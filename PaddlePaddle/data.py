from random import shuffle, choice
import numpy as np
from tqdm import tqdm
import time
import ipdb

class DataLoader(object):
    def __init__(self, ori_data_path, dst_data_path, dist_data_path, seed):
        ori_raw = list(map(lambda x: x.strip().split('\t'), open(ori_data_path, 'r')))
        dst_raw = list(map(lambda x: x.strip().split('\t'), open(dst_data_path, 'r')))
        if dist_data_path:
            dist_raw = list(map(lambda x: x.strip().split('\t'), open(dist_data_path, 'r')))
        else:
            dist_raw = []        

        self.ori_poi_idx = {}
        self.ori_idx_poi = {}
        self.dst_poi_idx = {}
        self.dst_idx_poi = {}
        self.tag_idx = {}

        self.o_tag_pois = {}
        self.d_tag_pois = {}
        self.o_poi_users = {}
        self.d_poi_users = {}

        for i in ori_raw:
            uid, timestamp, bid, std_tag = i
            if bid not in self.ori_poi_idx:
                self.ori_poi_idx[bid] = len(self.ori_poi_idx) + 1
                self.ori_idx_poi[self.ori_poi_idx[bid]] = bid
            if std_tag not in self.tag_idx:
                self.tag_idx[std_tag] = len(self.tag_idx) + 1
            if self.tag_idx[std_tag] not in self.o_tag_pois:
                self.o_tag_pois[self.tag_idx[std_tag]] = set()
            if self.ori_poi_idx[bid] not in self.o_poi_users:
                self.o_poi_users[self.ori_poi_idx[bid]] = set()
            self.o_tag_pois[self.tag_idx[std_tag]].add(self.ori_poi_idx[bid])
            
        for i in dst_raw:
            uid, timestamp, bid, std_tag = i
            if bid not in self.dst_poi_idx:
                self.dst_poi_idx[bid] = len(self.dst_poi_idx) + 1
                self.dst_idx_poi[self.dst_poi_idx[bid]] = bid
            if std_tag not in self.tag_idx:
                self.tag_idx[std_tag] = len(self.tag_idx) + 1
            if self.tag_idx[std_tag] not in self.d_tag_pois:
                self.d_tag_pois[self.tag_idx[std_tag]] = set()
            if self.dst_poi_idx[bid] not in self.d_poi_users:
                self.d_poi_users[self.dst_poi_idx[bid]] = set()
            self.d_tag_pois[self.tag_idx[std_tag]].add(self.dst_poi_idx[bid])

        self.dist = np.zeros((len(self.dst_poi_idx) + 1, len(self.dst_poi_idx) + 1), dtype=float)
        for i in dist_raw:
            start_poi, end_poi, poi_dist = i
            self.dist[self.dst_poi_idx[start_poi], self.dst_poi_idx[end_poi]] = np.exp(-float(poi_dist))

        self.oris = []
        self.dsts = []
        self.users = []
        
        ori_buffer = []
        dst_buffer = []

        last_uid = '0'
        for i in ori_raw:
            uid, timestamp, bid, std_tag = i
            if uid != last_uid:
                self.oris.append(ori_buffer)
                ori_buffer = []
                self.users.append(int(last_uid))
                last_uid = uid
            ori_buffer.append((self.ori_poi_idx[bid], self.tag_idx[std_tag], timestamp))
            self.o_poi_users[self.ori_poi_idx[bid]].add(int(uid))
        # the last
        self.users.append(int(last_uid))
        self.oris.append(ori_buffer)
        
        last_uid = '0'
        for i in dst_raw:
            uid, timestamp, bid, std_tag = i
            if uid != last_uid:
                self.dsts.append(dst_buffer)
                dst_buffer = []
                last_uid = uid
            dst_buffer.append((self.dst_poi_idx[bid], self.tag_idx[std_tag], timestamp))
        self.dsts.append(dst_buffer)

        shuf_idx = list(range(len(self.users)))
        shuffle(shuf_idx)
        self.oris = np.array(self.oris, dtype=object)[shuf_idx].tolist()
        self.dsts = np.array(self.dsts, dtype=object)[shuf_idx].tolist()
        self.users = np.array(self.users, dtype=object)[shuf_idx].tolist()
    
    def split_dataset(self, train_ratio=0.8, valid_ratio=0.1):
        train_num = int(len(self.oris) * train_ratio)
        valid_num = int(len(self.oris) * valid_ratio)

        self.train_ori = self.oris[:train_num]
        self.train_dst = self.dsts[:train_num]
        self.train_users = self.users[:train_num]

        self.valid_ori = self.oris[train_num:train_num+valid_num]
        self.valid_dst = self.dsts[train_num:train_num+valid_num]
        self.valid_users = self.users[train_num:train_num+valid_num]

        self.test_ori = self.oris[train_num+valid_num:]
        self.test_dst = self.dsts[train_num+valid_num:]
        self.test_users = self.users[train_num+valid_num:]

    def ori_poi_num(self):
        return len(self.ori_poi_idx)
    
    def dst_poi_num(self):
        return len(self.dst_poi_idx)
    
    def user_num(self):
        return len(self.users)


class DataExc(object):
    def __init__(self, ori, dst, users):
        self.ori = ori
        self.dst = dst
        self.users = users
    
    def shuffle_data(self):
        shuf_idx = list(range(len(self.ori)))
        shuffle(shuf_idx)
        self.ori = np.array(self.ori, dtype=object)[shuf_idx].tolist()
        self.dst = np.array(self.dst, dtype=object)[shuf_idx].tolist()
        self.users = np.array(self.users, dtype=object)[shuf_idx].tolist()
    
    def get_batch(self, i, batch_size):
        ori_batch = self.ori[i * batch_size: (i + 1) * batch_size]
        dst_batch = self.dst[i * batch_size: (i + 1) * batch_size]
        users_batch = self.users[i * batch_size: (i + 1) * batch_size]
        return ori_batch, dst_batch, users_batch
        
    def batch_num(self, batch_size):   
        return int(len(self.ori) / batch_size) + 1