import os
import random
import numpy as np
import torch
import time
import pickle
from scipy import sparse
import ipdb

#import dgl

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(model, optimizer, scheduler, save_dir, i):
    """ save current model """
    torch.save({
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict()
        }, os.path.join(save_dir, 'model_%d.xhr' % i))

def path_exist(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def cal_hr(tops,labels):
    res = []
    flat_res = []
    for tk in [10,20,30]:
        recall = []
        for _,(top, label) in enumerate(zip(tops, labels)):
            r = 0.
            for k in top[:tk]:
                r += np.isin(k,label)
            r /= len(set(label))
            recall.append(r)
        flat_res.append(recall)
        recall = np.mean(recall) * 100
        res.append(recall)
    return res[0], res[1], res[2], flat_res[0], flat_res[1], flat_res[2]

def cal_map(tops,labels):
    map_ = []
    for top,label in zip(tops, labels):
        m = 0.
        relative_num = 0.
        for i,k in enumerate(top,start=1):
            if np.isin(k,label):
                m += (relative_num + 1) / i
                relative_num += 1
        if relative_num > 0:
            m /= relative_num
        map_.append(m)
    return np.mean(map_) * 100, map_

class Logger(object):
    def __init__(self, log_path, name, seed, is_write_file=True):
        cur_time = time.strftime("%m-%d-%H:%M", time.localtime())
        self.is_write_file = is_write_file
        if self.is_write_file:
            self.log_file = open(os.path.join(log_path, "%s %s(%d).log" % (cur_time, name, seed)), 'w')
    
    def log(self, log_str):
        out_str = "[%s] " % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + log_str
        print(out_str)
        if self.is_write_file:
            self.log_file.write(out_str+'\n')
            self.log_file.flush()
    
    def close_log(self):
        if self.is_write_file:
            self.log_file.close()