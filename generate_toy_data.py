import random as rd
from tqdm import tqdm

user_num = 1000
ori_poi_num = 100
dst_poi_num = 100

ori_file = open("./toy_data/ori_checkin.txt", 'w')
dst_file = open("./toy_data/dst_checkin.txt", "w")
dist_file = open("./toy_data/dist.txt", 'w')

for u in tqdm(range(user_num)):
    ori_len = rd.randint(5, 20)
    dst_len = rd.randint(2, 10)
    ori_basetime = rd.randint(1577808000, 1606752000)
    dst_basetime = rd.randint(ori_basetime, 1622476800)

    for i in range(ori_len):
        timestamp = ori_basetime + rd.randint(900, 21600)
        ori_basetime = timestamp
        poi_id = rd.randint(0, ori_poi_num-1)
        tag = 'none'

        ori_file.write("{}\t{}\t{}\t{}\n".format(u, timestamp, poi_id, tag))
    
    for i in range(dst_len):
        timestamp = dst_basetime + rd.randint(900, 21600)
        dst_basetime = timestamp
        poi_id = rd.randint(0, dst_poi_num-1)
        tag = 'none'
        
        dst_file.write("{}\t{}\t{}\t{}\n".format(u, timestamp, poi_id, tag))

for i in tqdm(range(dst_poi_num)):
    for j in range(dst_poi_num):
        if i == j:
            dist = 0.
        else:
            dist = rd.uniform(100, 40000)
        dist_file.write("{}\t{}\t{}\n".format(i, j, dist))