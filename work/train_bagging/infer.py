import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.2"
from models import MLPClassifier
import numpy as np
import sys
import gc
import pandas as pd

test_path = pd.read_csv('../data_processing/get_basic_file/test.txt', names=['path'])

y = np.load('../data_processing/get_basic_file/y_train_44w.npy')
print(y.shape)
y = y - 1
print(y.min(), y.max())

tr_paths = [
'../feature_extracting/Basic_feature/train_basic_feature.npy',
'../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_holiday_visit_44w.npy',
'../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_holiday_user_44w.npy',
'../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_day.npy',
'../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_hour.npy',
'../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_hour_std.npy',
'../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_work_rest_fangjia_hour.npy',
'../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_work_rest_fangjia_day.npy',
'../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_hour_visit_44w.npy',
'../train_multimodel/train_mm_prob.npy',
'../feature_extracting/UserID_feature_global/train_global_feature.npy',
'../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_work_rest_fangjia_hour_std.npy',
'../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_hour_user_44w.npy',
]
te_paths = [
'../feature_extracting/Basic_feature/test_basic_feature.npy',
'../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_holiday_visit_44w.npy',
'../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_holiday_user_44w.npy',
'../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_day.npy',
'../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_hour.npy',
'../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_hour_std.npy',
'../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_work_rest_fangjia_hour.npy',
'../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_work_rest_fangjia_day.npy',
'../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_hour_visit_44w.npy',
'../train_multimodel/test_mm_prob.npy',
'../feature_extracting/UserID_feature_global/test_global_feature.npy',
'../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_work_rest_fangjia_hour_std.npy',
'../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_hour_user_44w.npy',
]

print('load train data... (used for calculate scaling range.)')

# 5,278
i = 0
x = np.zeros((400000, 5278))
for p in tr_paths:
    t = np.load(p)
    x[:, i:i+t.shape[1]] = np.array(t)
    i += t.shape[1]
    del t
    print(i)

print('cal scaling min&max')

min_ = x.min(axis=0)
max_ = x.max(axis=0)
max_min = max_ - min_

# for i in range(x.shape[1]):
#     if max_min[i] == 0:
#         x[:, i] = 0
#     else:
#         x[:, i] = (x[:, i] - min_[i]) / max_min[i]
#     sys.stdout.write('\r>> Processing %d / %d' % (i, x.shape[1]))
#     sys.stdout.flush()

del x

print('load test data...')

i = 0
test_x = np.zeros((100000, 5278))
for p in te_paths:
    t = np.load(p)
    test_x[:, i:i+t.shape[1]] = np.array(t)
    i += t.shape[1]
    del t
    print(i)

print('scaling...')

for i in range(test_x.shape[1]):
    if max_min[i] == 0:
        test_x[:, i] = 0
    else:
        test_x[:, i] = (test_x[:, i] - min_[i]) / max_min[i]
    sys.stdout.write('\r>> Processing %d / %d' % (i, test_x.shape[1]))
    sys.stdout.flush()
    
print(min_, max_)

print(test_x.max(), test_x.min())

def data_reader(a, b, idx, fidx):
    def reader():
        for i in idx:
            a_, b_ = a[i], b[i]
            a_ = a_.T[fidx].T
            yield a_, b_
    return reader

test_y = np.zeros((len(test_x), 9))
test_path['AreaID'] = test_path['path'].apply(lambda x: x.split('/')[-1].split('.')[0])

# for i in range(50):
for i in range(46):

    print('-' * 80)
    print('Sample %d:' % i)
    
    fidx = np.load('./tmp/sample_%d/fidx.npy' % i)
    print(fidx.shape)

    model = MLPClassifier(input_dim=len(fidx), cls_num=9, hidden_layer_sizes=(256, 128, 64), lr=1e-5)
    
    model.load_model('./tmp/sample_%d/best_model' % i)

    print('infer...')

    test_reader = data_reader(test_x, np.zeros(len(test_x)), np.arange(100000), fidx)
    pred = model.predict_prob(test_reader, batch_size=1024)
    
    test_y += pred
    
    print('infer done.')

np.save('./test_prob_bagging.npy', test_y)
test_y = np.argmax(test_y, axis=1) + 1
print(test_y.min(), test_y.max())

test_path['CategoryID'] = test_y
test_path['CategoryID'] = test_path['CategoryID'].apply(lambda x: '%03d' % x)

print(test_path.head())

test_path[['AreaID', 'CategoryID']].to_csv('./result_bagging.txt', index=False, header=False, sep='\t')
test_path[['AreaID', 'CategoryID']].to_csv('~/result.txt', index=False, header=False, sep='\t')

print('All Done.')