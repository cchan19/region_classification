import numpy as np
import pandas as pd

trn_prob = np.load('train_mm_prob.npy')
test_prob = np.load('test_mm_prob.npy')

y = np.load('../data_processing/get_basic_file/y_train_44w.npy')
print(y.shape)
y = y - 1
print(y.min(), y.max())

acc = ((y - np.argmax(trn_prob, axis=1)) == 0).sum() / len(y)

print('acc: %f' % acc)

test_path = pd.read_csv('../data_processing/get_basic_file/test.txt', names=['path'])

test_path['AreaID'] = test_path['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
test_path['CategoryID'] = np.argmax(test_prob, axis=1) + 1
test_path['CategoryID'] = test_path['CategoryID'].apply(lambda x: '%03d' % x)

print(test_path.head())

test_path[['AreaID', 'CategoryID']].to_csv('./result_stack.txt', index=False, header=False, sep='\t')