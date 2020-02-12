from sklearn.datasets import load_iris
from tqdm import trange
import numpy as np
import random
import json
import os
# Setup directory for train/test data
train_path = './data/train/iris.json'
test_path = './data/test/iris.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
# Get MNIST data, normalize, and divide by level
iris = load_iris()
mu = np.mean(iris.data.astype(np.float32), 0)
sigma = np.std(iris.data.astype(np.float32), 0)
iris.data = (iris.data.astype(np.float32) - mu)/(sigma+0.001)
iris_data = []
for i in trange(3):
    idx = iris.target==i
    iris_data.append(iris.data[idx])

print([len(v) for v in iris_data])

###### CREATE USER DATA SPLIT #######
# Assign n samples to each user
n=30
X = [[] for _ in range(5)]
y = [[] for _ in range(5)]
idx = np.zeros(n, dtype=np.int64)
for user in range(5):
    for j in range(2):
        l = (user+j)%3
        X[user] += iris_data[l][idx[l]:idx[l]+int(n/2)].tolist()
        y[user] += (l*np.ones(int(n/2))).tolist()
        idx[l] += int(n/2)

# Assign remaining sample by power law
user = 0
props = np.random.lognormal(0, 2.0, (3,5,2))
props = np.array([[[len(v)-5]] for v in iris_data])*props/np.sum(props,(1,2), keepdims=True)
#idx = 1000*np.ones(10, dtype=np.int64)
for user in trange(5):
    for j in range(2):
        l = (user+j)%3
        num_samples = int(props[l,user//3,j])
        #print(num_samples)
        if idx[l] + num_samples < len(iris_data[l]):
            X[user] += iris_data[l][idx[l]:idx[l]+num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples

# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 1000 users
for i in trange(5, ncols=120):
    uname = 'f_{0:05d}'.format(i)
    combined = list(zip(X[i], y[i]))
    if len(X[i])!=0 and len(y[i])!=0:
      random.shuffle(combined)
      X[i][:], y[i][:] = zip(*combined)
      num_samples = len(X[i])
      train_len = int(0.9*num_samples)
      test_len = num_samples - train_len

      train_data['users'].append(uname) 
      train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
      train_data['num_samples'].append(train_len)
      test_data['users'].append(uname)
      test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
      test_data['num_samples'].append(test_len)


with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)