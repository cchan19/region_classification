import os
import sys
import pandas as pd
main_data_path = '../../../data/'

train_visit_path = main_data_path + 'train_visit/'

ID = []
LABEL = []
Files = os.listdir(train_visit_path)
file_num = len(Files)
print('file_num:', file_num)
for index, file in enumerate(Files):
    Id = file[:file.find('.')]
    label = int(Id[-1]) - 1
    ID.append(Id)
    LABEL.append(label)

    sys.stdout.write(
        '\r>> Processing visit data %d/%d --- file name: %s -- Id: %s , label: %d' % (index + 1, file_num, file, Id, label))
    sys.stdout.flush()

sys.stdout.write("\n")

data = {'Id': ID, 'Target': LABEL}
data = pd.DataFrame.from_dict(data)
data.to_csv('train_44w.csv', index=False)
print('done!')

