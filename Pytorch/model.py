import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
import ipdb
import random
from tqdm import tqdm

import dgl
from dgl.nn.pytorch import GATConv, RelGraphConv

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GGNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class GCN(nn.Module):
    def __init__(self, hidden_size):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolutionLayer(hidden_size, hidden_size)
        self.dropout = nn.Dropout()
        self.act = nn.ReLU()

    def forward(self, x, adj):
        x = self.act(self.gc1(x, adj))
        return x


class NTM(nn.Module):
    def __init__(self, hidden_size, topic_num, bow_size):
        super(NTM,self).__init__()
        self.topic = nn.Embedding(topic_num,hidden_size)
        self.word = nn.Embedding(bow_size,hidden_size)

        self.enc_1 = nn.Linear(bow_size, hidden_size)
        self.enc_2 = nn.Linear(hidden_size, hidden_size)
        self.enc_shortcut = nn.Linear(bow_size, hidden_size, bias=False)

        self.mean_linear = nn.Linear(hidden_size, topic_num)
        self.mean_bn = nn.BatchNorm1d(topic_num)
        self.logsigma_linear = nn.Linear(hidden_size, topic_num)
        self.logsigma_bn = nn.BatchNorm1d(topic_num)

        self.gsm_enc_linear = nn.Linear(topic_num, topic_num)

    def forward(self, bow):
        # in town 
        ## encoder
        #en1 = F.softplus(dst_emb)
        en1 = torch.relu(self.enc_1(bow))
        enc_vec = torch.relu(self.enc_2(en1))
        
        mean = self.mean_linear(enc_vec)
        logsigma = self.logsigma_linear(enc_vec)

        kld = - 0.5 * torch.sum(1 - mean ** 2 + 2 * logsigma - torch.exp(2 * logsigma), dim = 1)

        epsilon = torch.zeros_like(mean).normal_()
        z = mean + torch.mul(epsilon, torch.exp(logsigma / 2))

        theta = F.softmax(self.gsm_enc_linear(z),dim=-1)
        ## decoder
        phi = F.softmax(torch.matmul(self.topic.weight, self.word.weight.transpose(0,1)), dim=-1)
        word_topic = torch.softmax(torch.matmul(self.word.weight, self.topic.weight.transpose(0, 1)), dim=-1)

        w_n = torch.matmul(theta, phi)
        reconstruction = - torch.sum(bow * torch.log(w_n + 1e-32), dim=1)
        loss = reconstruction + kld

        loss = torch.mean(loss)
        return loss, self.topic.weight, theta, phi, word_topic


class TRAINOR(nn.Module):
    def __init__(self, opt, n_ori_poi, n_dst_poi, n_user, sample_size, graph, device):
        super(TRAINOR, self).__init__()
        self.device = device
        self.hidden_size = opt.hidden_size
        self.n_ori_poi = n_ori_poi + 1 # 1 for the padding token
        self.n_dst_poi = n_dst_poi + 1
        self.n_user = n_user
        self.sample_size = sample_size
        self.args = opt

        self.graph = torch.Tensor(graph).to(self.device)

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

        self.softmax = nn.Softmax(dim=-1)
        self.linear_dst_w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.d_u_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def gated_gnn(self, it_visits, ori_len, o_poi_emb):
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
            #ipdb.set_trace()
            #u_A_in = u_A / u_sum_in
            u_A_in = np.divide(u_A, u_sum_in)
            #u_A_in = (torch.from_numpy(u_A) / torch.from_numpy(u_sum_in)).numpy()
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            #u_A_out = (torch.from_numpy(u_A.transpose()) / torch.from_numpy(u_sum_out)).numpy()
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            #u_A_out = u_A.transpose(0, 1) / u_sum_out
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            #u_A = torch.cat([u_A_in, u_A_out]).transpose(0, 1)
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) # re-index
        alias = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(A).to(self.device)
        #A = torch.cat(A, dim=0).to(self.device)
        items = torch.LongTensor(items).to(self.device)
        
        # GNN
        items_emb = o_poi_emb(items)
        hidden = self.gnn(A,items_emb)
        it_hidden = torch.stack([hidden[i][alias[i]] for i in torch.arange(len(alias)).long()])

        it_mask = torch.where(it_visits != 0,torch.ones_like(it_visits),it_visits)
        q = self.linear_q(it_hidden)
        alpha = self.linear_attn(torch.sigmoid(q))
        a = torch.sum(alpha * it_hidden * it_mask.view(it_mask.shape[0], -1, 1).float(), 1) # s_g(global)

        return a

    def preprocess(self, batch, is_train=True):
        # (bid, std_tag, timestamp)
        ori, dst, users = batch

        ori_checkin = list(map(lambda x: torch.tensor(list(map(lambda y: y[0], x))).to(self.device), ori))
        dst_checkin = list(map(lambda x: torch.tensor(list(map(lambda y: y[0], x))).to(self.device), dst))
        users = torch.LongTensor(users).to(self.device)

        dst_bow = torch.stack([torch.bincount(torch.unique(x),minlength=self.n_dst_poi) for x in dst_checkin]).float()

        return ori_checkin, dst_checkin, users, dst_bow
    
    def sample(self, score, truth, region='ori'):
        index_row, pos_index_col, neg_index_col = [], [], []
        n_poi = self.n_ori_poi if region == 'ori' else self.n_dst_poi
        for idx, t in enumerate(truth):
            t = t.cpu().numpy()
            pos_sample = np.random.choice(t, self.sample_size)
            pos_index_col.append(pos_sample)
            neg_sample = np.random.choice(np.setdiff1d(np.arange(n_poi), t), self.sample_size)
            neg_index_col.append(neg_sample)
            index_row.append([idx] * self.sample_size)
        positive_tensor = score[index_row, pos_index_col]
        negative_tensor = score[index_row, neg_index_col]
        return positive_tensor, negative_tensor

    def forward(self, batch, is_train=True):
        ori_checkin, dst_checkin, batch_users, dst_bow = self.preprocess(batch)

        dst_ntm_loss, dst_memory, dst_theta, dst_phi, dst_wt = self.dst_ntm(dst_bow)

        pad_ori_checkin = pad_sequence(ori_checkin, batch_first=True) # B x L
        
        ori_len = torch.LongTensor([i.size(0) for i in ori_checkin]).to(self.device)
        ori_preference = self.gated_gnn(pad_ori_checkin, ori_len, self.ori_poi_embedding) # B x H

        dst_preference = self.dst_user_embedding(batch_users) # B x H
        dst_preference_hat = self.mlp(ori_preference)
        dst_poi_emb = self.gcn(self.dst_poi_embedding.weight, self.graph)

        dst_alpha = torch.softmax(torch.matmul(self.linear_dst_w1(dst_memory), ori_preference.unsqueeze(2)), dim=-2)
        dst_intent = torch.matmul(dst_alpha.transpose(1, 2), dst_memory).squeeze(1)

        pred = torch.matmul(self.d_u_fusion(torch.cat([dst_intent, dst_preference_hat], dim=-1)), dst_poi_emb.transpose(0,1)) # B x M (test)
        score = torch.matmul(self.d_u_fusion(torch.cat([dst_intent, dst_preference], dim=-1)), dst_poi_emb.transpose(0,1)) # B x M
        positive_score, negative_score = self.sample(score, dst_checkin, 'dst') # B x S

        bpr_loss = - torch.mean(F.logsigmoid(positive_score - negative_score))
        transfer_loss = torch.mean((dst_preference_hat - dst_preference) ** 2)

        var = 0.9
        loss = (1 - var) / 2 * bpr_loss + (1 - var) / 2 * transfer_loss + var * dst_ntm_loss
        return loss, pred