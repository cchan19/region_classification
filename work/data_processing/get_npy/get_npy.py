import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
import datetime

# visit2array 方法定义

# 用字典查询代替类型转换，可以减少一部分计算时间
date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i

# 访问记录内的时间从 2018年10月1日 到 2019年3月31日，共182天
# 将日期按日历排列
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
    date_int = int(date.__str__().replace("-", ""))
    date2position[date_int] = [i%7, i//7]
    datestr2dateint[str(date_int)] = date_int

## main
# 转换 train_visit 数据
train_visit_path = '../../../data/train_visit/'
test_visit_path = '../../../data/test_visit/'
train_npy_path = "./npy/train_visit/"
test_npy_path = "./npy/test_visit/"
if not os.path.exists(test_npy_path):
    os.makedirs(test_npy_path)
if not os.path.exists(train_npy_path):
    os.makedirs(train_npy_path)

## train

visit_path_filelist = os.listdir(train_visit_path)
print(len(visit_path_filelist))
visit_path_filelist = [visit_path_filelist[i*50000:(i+1)*50000] for i in range(8)]
print(len(visit_path_filelist))

def visit2array(filenames):
    path = train_visit_path
    savepath = train_npy_path
    # filenames = os.listdir(path)
    cnt = 0
    length = len(filenames)
    for index, filename in enumerate(filenames):
        table = pd.read_table(os.path.join(path, filename), header=None)
        strings = table[1]
        init = np.zeros((7, 26, 24))
        for string in strings:
            temp = []
            for item in string.split(','):
                temp.append([item[0:8], item[9:].split("|")])
            for date, visit_lst in temp:
                # x - 第几天
                # y - 第几周
                # z - 几点钟
                # value - 到访的总人数
                x, y = date2position[datestr2dateint[date]]
                for visit in visit_lst: # 统计到访的总人数
                    init[x][y][str2int[visit]] += 1
        np.save(savepath + "/"+filename.split('.')[0]+".npy", init)
        print('Processing visit data %d/%d' % (index+1, length), end='\r')
        cnt += 1
    return cnt

with Pool(8) as pool:
    pool.map(visit2array, visit_path_filelist)

## test

visit_path_filelist = os.listdir(test_visit_path)
print(len(visit_path_filelist))
visit_path_filelist = [visit_path_filelist[i*12500:(i+1)*12500] for i in range(8)]
print(len(visit_path_filelist))

def visit2array(filenames):
    path = test_visit_path
    savepath = test_npy_path
    # filenames = os.listdir(path)
    cnt = 0
    length = len(filenames)
    for index, filename in enumerate(filenames):
        table = pd.read_table(os.path.join(path, filename), header=None)
        strings = table[1]
        init = np.zeros((7, 26, 24))
        for string in strings:
            temp = []
            for item in string.split(','):
                temp.append([item[0:8], item[9:].split("|")])
            for date, visit_lst in temp:
                # x - 第几天
                # y - 第几周
                # z - 几点钟
                # value - 到访的总人数
                x, y = date2position[datestr2dateint[date]]
                for visit in visit_lst: # 统计到访的总人数
                    init[x][y][str2int[visit]] += 1
        np.save(savepath + "/"+filename.split('.')[0]+".npy", init)
        print('Processing visit data %d/%d' % (index+1, length), end='\r')
        cnt += 1
    return cnt

with Pool(8) as pool:
    pool.map(visit2array, visit_path_filelist)