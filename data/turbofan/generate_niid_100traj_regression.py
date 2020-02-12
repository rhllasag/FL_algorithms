from tqdm import trange
import numpy as np
import random
import json
import os
import pandas as pd
from sklearn import preprocessing

train_path = './data/train/turbofan_train.json'
test_path = './data/test/turbofan_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get TURBIFAN data, normalize, and divide by trajectory
train_df = pd.read_csv('data/PM_train.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id','cycle'])

# read test data - It is the aircraft engine operating data without failure events recorded.
test_df = pd.read_csv('data/PM_test.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']

# read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
truth_df = pd.read_csv('data/PM_truth.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

#######
# TRAIN
#######
# Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)
# generate RUL for test data
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)
#Normalize columns in Train data exept the listed below
cols_normalize = train_df.columns.difference(['id','cycle'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df) 
train_df = join_df.reindex(columns = train_df.columns)
#Normalize columns in Test data using the values of train data
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)
# Naming columns to training the model 
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3']
sequence_cols.extend(sensor_cols)

#Group data by id trajectory
routes_train = {}
for unit_nr in train_df['id'].unique():
    routes_train[unit_nr-1] = train_df.loc[train_df['id'] == unit_nr]
routes_test = {}
for unit_nr in test_df['id'].unique():
    routes_test[unit_nr-1] = test_df.loc[test_df['id'] == unit_nr]

X_train = [[] for _ in range(100)]
y_train = [[] for _ in range(100)]
X_test = [[] for _ in range(100)]
y_test = [[] for _ in range(100)]

for i in range(0,100):
  X_train[i]=routes_train[i][sequence_cols].values.tolist()
  y_train[i]=routes_train[i]['RUL'].T.to_numpy().tolist()
  X_test[i]=routes_test[i][sequence_cols].values.tolist()
  y_test[i]=routes_test[i]['RUL'].T.to_numpy().tolist()
# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
for i in trange(100, ncols=150):
    uname = 'f_{0:05d}'.format(i)
    combined_train = list(zip(X_train[i], y_train[i]))
    combined_test = list(zip(X_test[i], y_test[i]))
    if len(X_train[i])!=0 and len(y_train[i])!=0 and len(X_test[i])!=0 and len(y_test[i])!=0:
      random.shuffle(combined_train)
      random.shuffle(combined_test)
      X_train[i][:], y_train[i][:] = zip(*combined_train)
      X_test[i][:], y_test[i][:] = zip(*combined_test)
      num_samples_train = len(X_train[i])
      num_samples_test = len(X_test[i])
      train_data['users'].append(uname) 
      train_data['user_data'][uname] = {'x': X_train[i][:num_samples_train], 'y': y_train[i][:num_samples_train]}
      train_data['num_samples'].append(num_samples_train)
      test_data['users'].append(uname)
      test_data['user_data'][uname] = {'x': X_test[i][:num_samples_test], 'y': y_test[i][:num_samples_test]}
      test_data['num_samples'].append(num_samples_test)

with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)