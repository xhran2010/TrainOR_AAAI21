import datetime
import math
import numpy as np
import ipdb
import random
from tqdm import tqdm

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import create_parameter


class GraphConvolutionLayer(nn.Layer):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = create_parameter(shape=[in_features, out_features], dtype='float32')
        if bias:
            self.bias = create_parameter(shape=[out_features], dtype='float32')

    def forward(self, input, adj):
        support = paddle.mm(input, self.weight)
        output = paddle.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GGNN(nn.Layer):
    def __init__(self, hidden_size, step=1):
        super(GGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        
        self.linear_ih = nn.Linear(self.input_size, self.gate_size)
        self.linear_hh = nn.Linear(self.hidden_size, self.gate_size)
        self.b_iah = create_parameter(shape=[self.hidden_size], dtype='float32')
        self.b_oah = create_parameter(shape=[self.hidden_size], dtype='float32')

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size)

    def GNNCell(self, A, hidden):
        input_in = paddle.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = paddle.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = paddle.concat([input_in, input_out], axis=2)
        gi = self.linear_ih(inputs)
        gh = self.linear_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class GCN(nn.Layer):
    def __init__(self, hidden_size):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolutionLayer(hidden_size, hidden_size)
        self.dropout = nn.Dropout()
        self.act = nn.ReLU()

    def forward(self, x, adj):
        x = self.act(self.gc1(x, adj))
        return x


class NTM(nn.Layer):
    def __init__(self, hidden_size, topic_num, bow_size):
        super(NTM,self).__init__()
        self.topic = nn.Embedding(topic_num,hidden_size)
        self.word = nn.Embedding(bow_size,hidden_size)

        self.enc_1 = nn.Linear(bow_size, hidden_size)
        self.enc_2 = nn.Linear(hidden_size, hidden_size)
        self.enc_shortcut = nn.Linear(bow_size, hidden_size, bias_attr=False)

        self.mean_linear = nn.Linear(hidden_size, topic_num)
        self.mean_bn = nn.BatchNorm1D(topic_num)
        self.logsigma_linear = nn.Linear(hidden_size, topic_num)
        self.logsigma_bn = nn.BatchNorm1D(topic_num)

        self.gsm_enc_linear = nn.Linear(topic_num, topic_num)

    def forward(self, bow):
        # in town 
        ## encoder
        #en1 = F.softplus(dst_emb)
        en1 = F.relu(self.enc_1(bow))
        enc_vec = F.relu(self.enc_2(en1))
        
        mean = self.mean_linear(enc_vec)
        logsigma = self.logsigma_linear(enc_vec)

        kld = - 0.5 * paddle.sum(1 - mean ** 2 + 2 * logsigma - paddle.exp(2 * logsigma), axis = 1)

        # epsilon = paddle.zeros_like(mean).normal_()
        epsilon = paddle.normal(shape=mean.shape)
        z = mean + epsilon * paddle.exp(logsigma / 2)

        theta = F.softmax(self.gsm_enc_linear(z),axis=-1)
        ## decoder
        phi = F.softmax(paddle.matmul(self.topic.weight, self.word.weight.transpose([1,0])), axis=-1)
        word_topic = F.softmax(paddle.matmul(self.word.weight, self.topic.weight.transpose([1,0])), axis=-1)

        w_n = paddle.matmul(theta, phi)
        reconstruction = - paddle.sum(bow * paddle.log(w_n + 1e-32), axis=1)
        loss = reconstruction + kld

        loss = paddle.mean(loss)
        return loss, self.topic.weight, theta, phi, word_topic


class TRAINOR(nn.Layer):
    def __init__(self, opt, n_ori_poi, n_dst_poi, n_user, sample_size, graph):
        super(TRAINOR, self).__init__()
        self.hidden_size = opt.hidden_size
        self.n_ori_poi = n_ori_poi + 1 # 1 for the padding token
        self.n_dst_poi = n_dst_poi + 1
        self.n_user = n_user
        self.sample_size = sample_size
        self.args = opt

        self.graph = paddle.to_tensor(graph, dtype='float32')

        nn.initializer.set_global_initializer(nn.initializer.XavierUniform())

        self.ori_poi_embedding = nn.Embedding(self.n_ori_poi, self.hidden_size)
        self.dst_poi_embedding = nn.Embedding(self.n_dst_poi, self.hidden_size)
        self.dst_user_embedding = nn.Embedding(self.n_user, self.hidden_size)
        
        self.gnn = GGNN(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
        self.gcn = GCN(self.hidden_size)
        self.dst_ntm = NTM(self.hidden_size, opt.dst_topic, self.n_dst_poi)

        self.linear_q = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_attn = nn.Linear(self.hidden_size,1)

        self.softmax = nn.Softmax(axis=-1)
        self.logsigmoid = nn.LogSigmoid()
        self.linear_dst_w1 = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False)

        self.d_u_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )

    def gated_gnn(self, it_visits, ori_len, o_poi_emb, aggregate=True):
        # pre to graph structure
        inputs = it_visits.cpu().numpy()
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0) # out degree of every u
            u_sum_in[np.where(u_sum_in == 0)] = 1 # avoid divided by zero
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) # re-index
        alias = paddle.to_tensor(alias_inputs)
        A = paddle.to_tensor(A)
        items = paddle.to_tensor(items)
        
        # GNN
        items_emb = o_poi_emb(items)
        hidden = self.gnn(A,items_emb)
        it_hidden = paddle.stack([hidden[i][alias[i]] for i in paddle.arange(len(alias))])

        # traditional attention
        it_mask = paddle.where(it_visits != 0,paddle.ones_like(it_visits),it_visits)
        q = self.linear_q(it_hidden)
        alpha = self.linear_attn(F.sigmoid(q))
        a = paddle.sum(alpha * it_hidden * it_mask.reshape([it_mask.shape[0], -1, 1]), axis=1)

        return a

    def preprocess(self, batch, is_train=True):
        # (bid, std_tag, timestamp)
        ori, dst, users = batch

        ori_checkin = list(map(lambda x: paddle.to_tensor(list(map(lambda y: y[0], x))), ori))
        dst_checkin = list(map(lambda x: paddle.to_tensor(list(map(lambda y: y[0], x))), dst))
        users = paddle.to_tensor(users)

        dst_bow = np.stack([np.bincount(np.unique(x.numpy()),minlength=self.n_dst_poi) for x in dst_checkin])
        dst_bow = paddle.to_tensor(dst_bow, dtype='float32')

        return ori_checkin, dst_checkin, users, dst_bow
    
    def sample(self, score, truth, region='ori'):
        pos_list, neg_list = [], []
        n_poi = self.n_ori_poi if region == 'ori' else self.n_dst_poi
        for idx, t in enumerate(truth):
            t = t.cpu().numpy()
            pos_sample = np.random.choice(t, self.sample_size)
            pos_list.append(paddle.gather(score[idx], paddle.to_tensor(pos_sample)))
            neg_sample = np.random.choice(np.setdiff1d(np.arange(n_poi), t), self.sample_size)
            neg_list.append(paddle.gather(score[idx], paddle.to_tensor(neg_sample)))
        positive_tensor = paddle.stack(pos_list, axis=0)
        negative_tensor = paddle.stack(neg_list, axis=0)
        return positive_tensor, negative_tensor

    def forward(self, batch, is_train=True):
        ori_checkin, dst_checkin, batch_users, dst_bow = self.preprocess(batch)

        dst_ntm_loss, dst_memory, dst_theta, dst_phi, dst_wt = self.dst_ntm(dst_bow)

        ori_max_len = np.max([i.shape[0] for i in ori_checkin])

        pad_ori_checkin = paddle.stack([paddle.concat([i, paddle.zeros([ori_max_len - i.shape[0]], dtype='int64')]) for i in ori_checkin], axis=0)
        
        ori_len = paddle.to_tensor([i.shape[0] for i in ori_checkin])
        ori_preference = self.gated_gnn(pad_ori_checkin, ori_len, self.ori_poi_embedding, aggregate=True) # B x H

        dst_preference = self.dst_user_embedding(batch_users) # B x H
        dst_preference_hat = self.mlp(ori_preference)
        dst_poi_emb = self.gcn(self.dst_poi_embedding.weight, self.graph)

        dst_alpha = F.softmax(paddle.matmul(self.linear_dst_w1(dst_memory), dst_preference_hat.unsqueeze(2)), axis=-2)
        dst_intent = paddle.matmul(dst_alpha.transpose([0,2,1]), dst_memory).squeeze(1)

        pred = paddle.matmul(self.d_u_fusion(paddle.concat([dst_intent, dst_preference_hat], axis=-1)), dst_poi_emb.transpose([1,0])) # B x M (test)
        score = paddle.matmul(self.d_u_fusion(paddle.concat([dst_intent, dst_preference], axis=-1)), dst_poi_emb.transpose([1,0])) # B x M
        positive_score, negative_score = self.sample(score, dst_checkin, 'dst') # B x S

        bpr_loss = - paddle.mean(self.logsigmoid(positive_score - negative_score))
        transfer_loss = paddle.mean((dst_preference_hat - dst_preference) ** 2)

        var = 0.9
        loss = (1 - var) / 2 * bpr_loss + (1 - var) / 2 * transfer_loss + var * dst_ntm_loss

        return loss, pred