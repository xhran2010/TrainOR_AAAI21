# -*- coding: utf-8 -*-
import argparse
from collections import namedtuple
import numpy as np
import ipdb
import os
import sys
import time
from tqdm import tqdm
import pickle
from scipy.sparse import coo_matrix

from model import TRAINOR
from utils import set_seeds, save_model, cal_hr, cal_map, Logger, path_exist
from data import DataLoader, DataExc

import paddle
from paddle.optimizer import Adam, lr

def train(model, train_exc, valid_exc, args, logger):
    scheduler = lr.StepDecay(learning_rate=args.lr, step_size=args.lr_dc_step, gamma=args.lr_dc)
    optimizer = Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=args.l2)

    logger.log("Start training...")
    for e in range(args.epoch):
        # train
        model.train() # train mode
        loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
        train_exc.shuffle_data()
        n_train_batch = train_exc.batch_num(args.train_batch_size)
        hr10_list, hr20_list, hr30_list = [], [], []
        for i in tqdm(range(n_train_batch)):
            batch = train_exc.get_batch(i, args.train_batch_size)
            loss, score = model(batch, is_train=True)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_sum += loss.item()
        
        model.eval()
        for i in tqdm(range(n_train_batch)):
            batch = train_exc.get_batch(i, args.test_batch_size)
            loss, score = model.forward(batch, is_train=False)

            _, top30 = score.topk(k=30,axis=1,largest=True)
            top30 = top30.cpu().detach().numpy() # B x 30
            label = list(map(lambda x: list(map(lambda y: y[0], x)), batch[1]))
            hr_10, hr_20, hr_30, _, _, _ = cal_hr(top30, label)
            hr10_list.append(hr_10)
            hr20_list.append(hr_20)
            hr30_list.append(hr_30)

        logger.log("Epoch %d/%d : Average Train Loss %.10f, HR@10:%5.3f, HR@20: %5.3f, HR@30: %5.3f" % (e+1, args.epoch, loss_sum / n_train_batch, np.mean(hr10_list), np.mean(hr20_list), np.mean(hr30_list)))

        if ((e+1) % args.save_step == 0 or e == 0) and e > -1:
            save_model(model, os.path.join(args.save_path, "%s-%d" % (args.name, args.seed)), e + 1)
        
        # validation
        model.eval()
        n_valid_batch = valid_exc.batch_num(args.test_batch_size)
        rec10_list, rec20_list, rec30_list = [], [], []
        for i in tqdm(range(n_valid_batch)):
            batch = valid_exc.get_batch(i, args.test_batch_size)
            _, score = model.forward(batch, is_train=False)
            #valid_loss_sum += loss.item()
            _, top30 = score.topk(k=30,axis=1,largest=True)
            top30 = top30.cpu().detach().numpy() # B x 30
            label = list(map(lambda x: list(map(lambda y: y[0], x)), batch[1]))
            recall_10, recall_20, recall_30, _, _, _ = cal_hr(top30, label)
            rec10_list.append(recall_10)
            rec20_list.append(recall_20)
            rec30_list.append(recall_30)
        logger.log("Epoch %d/%d : Eval Rec@10:%5.3f, Rec@20: %5.3f, Rec@30: %5.3f" % \
            (e+1, args.epoch, np.mean(rec10_list), np.mean(rec20_list), np.mean(rec30_list)))

def test(model, model_path, test_exc, args, logger):
    checkpoint = paddle.load(model_path) 
    model.set_state_dict(checkpoint)
    model.eval()
    test_loss_sum = 0.
    n_test_batch = test_exc.batch_num(args.test_batch_size)
    rec10_list, rec20_list, rec30_list= [], [], []
    logger.log('Start testing...') 
    for i in tqdm(range(n_test_batch)):
        batch = test_exc.get_batch(i, args.test_batch_size)
        _, score = model(batch, is_train=False)
        #valid_loss_sum += loss.item()
        _, top30 = score.topk(k=30,axis=1,largest=True)
        top30 = top30.cpu().detach().numpy() # B x 30
        label = list(map(lambda x: list(map(lambda y: y[0], x)), batch[1]))
        recall_10, recall_20, recall_30, flat_10, flat_20, flat_30 = cal_hr(top30, label)
        rec10_list.append(recall_10)
        rec20_list.append(recall_20)
        rec30_list.append(recall_30)

    logger.log("Average Test Loss %.10f" % (test_loss_sum / n_test_batch))
    logger.log("Rec@10:%5.3f, Rec@20: %5.3f, Rec@30: %5.3f, MAP: %5.3f" % \
        (np.mean(rec10_list), np.mean(rec20_list), np.mean(rec30_list), np.mean(map_list)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_data', type=str, help="the data path of hometown check-ins", required=True)
    parser.add_argument('--dst_data', type=str, help="the data path of out-of-town check-ins", required=True)
    parser.add_argument('--dist_data', type=str, help="the data path of POI geographical distance", required=True)
    parser.add_argument('--save_path', type=str, help="the model save path", required=True)
    parser.add_argument('--test_path', type=str, help="only useful when mode is test")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_batch_size', type=int, default=1000)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr_dc', type=float, default=0.2)
    parser.add_argument('--lr_dc_step', type=int, default=20)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=9875)
    parser.add_argument('--log_path', type=str, default='./log/')
    parser.add_argument('--log', type=int, default=1) # bool
    parser.add_argument('--name', type=str, default="default")
    parser.add_argument('--dst_topic', type=int, default=15)
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args()

    set_seeds(args.seed)
    path_exist(os.path.join(args.save_path, "%s-%d" % (args.name, args.seed)))

    logger = Logger(args.log_path, args.name, args.seed, args.log)
    logger.log("Loading dataset...")

    paddle.set_device('gpu')

    data = DataLoader(args.ori_data, args.dst_data, args.dist_data, args.seed)
    data.split_dataset()

    train_exc = DataExc(data.train_ori, data.train_dst, data.train_users)
    valid_exc = DataExc(data.valid_ori, data.valid_dst, data.valid_users)
    test_exc = DataExc(data.test_ori, data.test_dst, data.test_users)

    model = TRAINOR(args, data.ori_poi_num(), data.dst_poi_num(), data.user_num(), args.sample_size, data.dist)

    if args.mode == 'train':
        train(model, train_exc, valid_exc, args, logger)
    elif args.mode == 'test':
        test(model, args.test_path, test_exc, args, logger)
    
    logger.close_log()
    
if __name__ == "__main__":
    main()