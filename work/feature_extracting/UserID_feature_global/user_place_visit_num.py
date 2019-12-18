# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import time
import os
from Config import config
from function_global_feature import get_global_feature_1, get_global_feature_2
from multiprocessing import Process, Pool


import pickle
data_list_path = '../../data_processing/get_basic_file/'

main_data_path = config.main_data_path

train_feature_out_path = config.train_feature_out_path  # './feature/train/'
train_table_path = config.train_table_path  # main_data_path + 'train.txt'
train_main_visit_path = config.train_main_visit_path  # main_data_path + "train_visit/train/"

test_feature_out_path = config.test_feature_out_path  # './feature/test/'
test_table_path = config.test_table_path  # main_data_path + 'test.txt'
test_main_visit_path = config.test_main_visit_path  # main_data_path + "test_visit/test/"

TEST_FLAG = False

train_num = 400000
test_num = 100000
# test_select_num = 10000
file_num_each_job_train = 10000
file_num_each_job_test = 2500
# file_num_each_job_test_select = 200
workers_train = int(train_num/file_num_each_job_train)
workers_test = int(test_num/file_num_each_job_test)

def static_user_place_num():
    user_place_visit_num = {}
    table = pd.read_csv(train_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    cnt_users = 0
    for index, filename in enumerate(filenames):
        table = pd.read_table(train_main_visit_path + filename + ".txt", header=None)
        label = int(filename.split("_")[1]) - 1
        users = table[0]
        cnt_users += len(users)
        # global_feature, len_feature = golbal_feature(table, num=num)
        for user in users:
            if user not in user_place_visit_num:
                user_place_visit_num[user] = []
            user_place_visit_num[user].append(index)
        sys.stdout.write(
            '\r>> Processing visit data %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print('totoal users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))

    return user_place_visit_num




if __name__ == '__main__':
    if not os.path.exists("./data/tmp/"):
        os.makedirs("./data/tmp/")
    user_place_visit_num = static_user_place_num()
    with open('./data/tmp/user_place_visit_num.pkl', 'wb') as file:
        pickle.dump(user_place_visit_num, file)
    print('Done')