from sklearn.datasets import fetch_mldata
from tqdm import trange
from balance import balanced_sample_maker
import numpy as np
import random
import json
import os

# Setup directory for train/test data
train_path = './data/train/all_data_0_niid_0_keep_10_train_9.json'
test_path = './data/test/all_data_0_niid_0_keep_10_test_9.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
# Get MNIST data, normalize, and divide by level
mnist = fetch_mldata('MNIST original', data_home='./data')
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)

X = [[] for _ in range(5)]
y = [[] for _ in range(5)]

#Loading data
x1_base = mnist.data
y1_base = mnist.target.reshape(-1,1) # Convert data to a single column

example_list=range(0,len(mnist.data))
exclude_list = []

#CLIENT 1 DATA
balanced_list1=balanced_sample_maker(x1_base,y1_base,20) # 1 is number per class 
X[0] = mnist.data[balanced_list1].tolist()
y[0] = mnist.target[balanced_list1].tolist()

#CLIENT 2 DATA
x2_base=mnist.data[list(set(example_list)-set(balanced_list1))]
y2_base=mnist.target[list(set(example_list)-set(balanced_list1))]

balanced_list2=balanced_sample_maker(x2_base,y2_base,30)

X[1] = mnist.data[balanced_list2].tolist()
y[1] = mnist.target[balanced_list2].tolist()
#CLIENT 3 DATA
x3_base=mnist.data[list(set(example_list)-set(balanced_list2))]
y3_base=mnist.target[list(set(example_list)-set(balanced_list2))]

balanced_list3=balanced_sample_maker(x3_base,y3_base,40)

X[2] = mnist.data[balanced_list3].tolist()
y[2] = mnist.target[balanced_list3].tolist()
#CLIENT 4 DATA
x4_base=mnist.data[list(set(example_list)-set(balanced_list3))]
y4_base=mnist.target[list(set(example_list)-set(balanced_list3))]

balanced_list4=balanced_sample_maker(x4_base,y4_base,50)

X[3] = mnist.data[balanced_list4].tolist()
y[3] = mnist.target[balanced_list4].tolist()

#CLIENT 5 DATA
x5_base=mnist.data[list(set(example_list)-set(balanced_list4))]
y5_base=mnist.target[list(set(example_list)-set(balanced_list4))]

balanced_list5=balanced_sample_maker(x5_base,y5_base,34)

X[4] = mnist.data[balanced_list5].tolist()
y[4] = mnist.target[balanced_list5].tolist()



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
      train_len = int(0.8*num_samples)
      test_len = num_samples - train_len

      train_data['users'].append(uname) 
      train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
      train_data['num_samples'].append(train_len)
      test_data['users'].append(uname)
      test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
      test_data['num_samples'].append(test_len)

#print(train_data)
print(train_data['num_samples'])
#print(test_data['user_data'])
print(test_data['num_samples'])

with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)