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

x1 = np.load('../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_holiday_visit_44w.npy')
x2 = np.load('../feature_extracting/UserID_feature_local/data/tmp/train_feature_user_holiday_user_44w.npy')
x3 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_day.npy')
x4 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_hour.npy')
x5 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_hour_std.npy')
x6 = np.load('../feature_extracting/UserID_feature_local/feature/train_X_UserID_normal_local_work_rest_fangjia_hour.npy')
x = np.hstack([x1, x2, x3, x4, x5, x6])
print(x.shape)

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
    
test_x1 = np.load('../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_holiday_visit_44w.npy')
test_x2 = np.load('../feature_extracting/UserID_feature_local/data/tmp/test_feature_user_holiday_user_44w.npy')
test_x3 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_day.npy')
test_x4 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_hour.npy')
test_x5 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_hour_std.npy')
test_x6 = np.load('../feature_extracting/UserID_feature_local/feature/test_X_UserID_normal_local_work_rest_fangjia_hour.npy')
test_x = np.hstack([test_x1, test_x2, test_x3, test_x4, test_x5, test_x6])
print(test_x.shape)

for i in range(test_x.shape[1]):
    if max_min[i] == 0:
        test_x[:, i] = 0
    else:
        test_x[:, i] = (test_x[:, i] - min_[i]) / max_min[i]
    sys.stdout.write('\r>> Processing %d / %d' % (i, test_x.shape[1]))
    sys.stdout.flush()
    
print(x.min(), x.max())

print(test_x.max(), test_x.min())

model = MLPClassifier(input_dim=x.shape[1], cls_num=9, hidden_layer_sizes=(256, 128, 64), lr=1e-5)

def data_reader(a, b):
    def reader():
        for i in range(len(a)):
            yield a[i], b[i]
    return reader

trn_x, val_x = x[:len(x)//2], x[len(x)//2:]
trn_y, val_y = y[:len(x)//2], y[len(x)//2:]

trn_reader = data_reader(trn_x, trn_y)
val_reader = data_reader(val_x, val_y)

model.fit(trn_reader=trn_reader, val_reader=val_reader, epoch_num=100, batch_size=256, model_dir='best_model')

model.load_model('best_model')

bs = model.score(val_reader, batch_size=1024)[0]
print(bs)

test_reader = data_reader(test_x, np.zeros(len(test_x)))
test_y = model.predict_prob(test_reader, batch_size=1024)
np.save('./test_prob.npy', test_y)
test_y = np.argmax(test_y, axis=1) + 1
print(test_y.min(), test_y.max())


test_path['AreaID'] = test_path['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
test_path['CategoryID'] = test_y
test_path['CategoryID'] = test_path['CategoryID'].apply(lambda x: '%03d' % x)

print(test_path.head())

test_path[['AreaID', 'CategoryID']].to_csv('./result.txt', index=False, header=False, sep='\t')