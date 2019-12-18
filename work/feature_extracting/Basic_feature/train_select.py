import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.2"
from models import MLPClassifier
import numpy as np
import sys
import gc
import pandas as pd

y = np.load('../../data_processing/get_basic_file/y_train_44w.npy')
print(y.shape)
y = y - 1
print(y.min(), y.max())

idx1 = np.load('./Code_Basic_feature_1/select_index.npy')
idx2 = np.load('./Code_Basic_feature_2/select_index.npy')
print('idx1 len: %d' % len(idx1))
print('idx2 len: %d' % len(idx2))

x1 = np.load('./Code_Basic_feature_1/feature/train_basic_feature_1_20.npy')[:, idx1]
x2 = np.load('./Code_Basic_feature_2/feature/train_basic_feature_2_20.npy')[:, idx2]
x = np.hstack([x1, x2])
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
    
test_x1 = np.load('./Code_Basic_feature_1/feature/test_basic_feature_1_20.npy')[:, idx1]
test_x2 = np.load('./Code_Basic_feature_2/feature/test_basic_feature_2_20.npy')[:, idx2]
test_x = np.hstack([test_x1, test_x2])
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

model.fit(trn_reader=trn_reader, val_reader=val_reader, epoch_num=40, batch_size=64, model_dir='best_model')

model.load_model('best_model')

bs = model.score(val_reader, batch_size=1024)[0]
print(bs)

print('select...')

scores = []
for i in range(val_x.shape[1]):
    arr = np.array(val_x[:, i])
    val_x[:, i] = 0
    val_reader = data_reader(val_x, val_y)
    sc = model.score(val_reader, batch_size=1024)[0]
    scores.append(sc)
    val_x[:, i] = arr
    sys.stdout.write('\r>> Processing %d / %d, score: %f' % (i, val_x.shape[1], sc))
    sys.stdout.flush()

scores = pd.DataFrame(scores, columns=['score'])
scores['index'] = np.arange(len(scores))

select_index = scores[scores['score'] < bs]['index'].values

print('select index: %d' % len(select_index))

np.save('select_index.npy', select_index)

print('Done.')

print('valid score')

trn_x, val_x = trn_x[:, select_index], val_x[:, select_index]
print(trn_x.shape)
trn_reader = data_reader(trn_x, trn_y)
val_reader = data_reader(val_x, val_y)
model = MLPClassifier(input_dim=trn_x.shape[1], cls_num=9, hidden_layer_sizes=(256, 128, 64), lr=1e-5)
model.fit(trn_reader=trn_reader, val_reader=val_reader, epoch_num=40, batch_size=64, model_dir='selected_model')