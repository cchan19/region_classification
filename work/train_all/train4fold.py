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

# 5278
i = 0
x = np.zeros((400000, 5278))
for p in tr_paths:
    t = np.load(p)
    x[:, i:i+t.shape[1]] = np.array(t)
    i += t.shape[1]
    del t
    print(i)

# ## 870
# x0 = np.load('../feature_extracting/Basic_feature/train_basic_feature.npy')
# x1 = np.load('../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_holiday_visit_44w.npy')
# x2 = np.load('../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_holiday_user_44w.npy')
# x3 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_day.npy')
# x4 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_hour.npy')
# x5 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_hour_std.npy')
# x6 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_work_rest_fangjia_hour.npy')
# ## 879
# x7 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_work_rest_fangjia_day.npy')
# x8 = np.load('../train_multimodel/train_mm_prob.npy')
# x9 = np.load('../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_hour_visit_44w.npy')
# ## 855
# x10 = np.load('../feature_extracting/UserID_feature_global/train_global_feature.npy')
# ## 
# x11 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_work_rest_fangjia_hour_std.npy')
# x = np.hstack([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])
# print(x.shape)

# del x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11
# gc.collect()

min_ = x.min(axis=0)
max_ = x.max(axis=0)
max_min = max_ - min_

for i in range(x.shape[1]):
    if max_min[i] == 0:
        x[:, i] = 0
    else:
        x[:, i] = (x[:, i] - min_[i]) / max_min[i]
    sys.stdout.write('\r>> Processing %d / %d' % (i, x.shape[1]))
    sys.stdout.flush()

i = 0
test_x = np.zeros((100000, 5278))
for p in te_paths:
    t = np.load(p)
    test_x[:, i:i+t.shape[1]] = np.array(t)
    i += t.shape[1]
    del t
    print(i)

# ## 
# test_x0 = np.load('../feature_extracting/Basic_feature/test_basic_feature.npy') 
# test_x1 = np.load('../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_holiday_visit_44w.npy')
# test_x2 = np.load('../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_holiday_user_44w.npy')
# test_x3 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_day.npy')
# test_x4 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_hour.npy')
# test_x5 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_hour_std.npy')
# test_x6 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_work_rest_fangjia_hour.npy')
# ##
# test_x7 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_work_rest_fangjia_day.npy')
# test_x8 = np.load('../train_multimodel/test_mm_prob.npy')
# test_x9 = np.load('../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_hour_visit_44w.npy')
# ##
# test_x10 = np.load('../feature_extracting/UserID_feature_global/test_global_feature.npy')
# ##
# x11 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_work_rest_fangjia_hour_std.npy')
# test_x = np.hstack([test_x0, test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_x7, test_x8, test_x9, test_x10, test_x11])
# print(test_x.shape)

# del test_x0, test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_x7, test_x8, test_x9, test_x10, test_x11
# gc.collect()

for i in range(test_x.shape[1]):
    if max_min[i] == 0:
        test_x[:, i] = 0
    else:
        test_x[:, i] = (test_x[:, i] - min_[i]) / max_min[i]
    sys.stdout.write('\r>> Processing %d / %d' % (i, test_x.shape[1]))
    sys.stdout.flush()
    
print(x.min(), x.max())

print(test_x.max(), test_x.min())

def data_reader(a, b, idx):
    def reader():
        for i in idx:
            yield a[i], b[i]
    return reader

kfold_idx = [(np.arange(100000, 400000), np.arange(0, 100000)),
             (np.hstack([np.arange(0, 100000), np.arange(200000, 400000)]), np.arange(100000, 200000)),
             (np.hstack([np.arange(0, 200000), np.arange(300000, 400000)]), np.arange(200000, 300000)),
             (np.arange(0, 300000), np.arange(300000, 400000))]

test_y = np.zeros((len(test_x), 9))

for k, (trn_idx, val_idx) in enumerate(kfold_idx):

    print('-' * 80)
    print('Fold %d:' % k)

    trn_reader = data_reader(x, y, trn_idx)
    val_reader = data_reader(x, y, val_idx)

    model = MLPClassifier(input_dim=x.shape[1], cls_num=9, hidden_layer_sizes=(256, 128, 64), lr=1e-5)

    model.fit(trn_reader=trn_reader, val_reader=val_reader, epoch_num=250, batch_size=256, model_dir='best_model_fold_%d' % k)

    model.load_model('best_model_fold_%d' % k)

    bs = model.score(val_reader, batch_size=1024)[0]
    print(bs)

    test_reader = data_reader(test_x, np.zeros(len(test_x)), np.arange(100000))
    test_y += model.predict_prob(test_reader, batch_size=1024) / 4

np.save('./test_prob_4fold.npy', test_y)
test_y = np.argmax(test_y, axis=1) + 1
print(test_y.min(), test_y.max())


test_path['AreaID'] = test_path['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
test_path['CategoryID'] = test_y
test_path['CategoryID'] = test_path['CategoryID'].apply(lambda x: '%03d' % x)

print(test_path.head())

test_path[['AreaID', 'CategoryID']].to_csv('./result_4fold.txt', index=False, header=False, sep='\t')