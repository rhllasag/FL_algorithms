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

example_list=range(0,150)
exclude_list = []

#First Node
valid_list = list(set(example_list))
first_node_examples=random.sample(valid_list,k=6)
#Second Node
exclude_list+=first_node_examples
valid_list = list(set(example_list) - set(exclude_list))
second_node_examples=random.sample(valid_list,k=7)
#Third Node 
exclude_list+=second_node_examples
valid_list = list(set(example_list) - set(exclude_list))
third_node_examples=random.sample(valid_list,k=8)
#Fourth Node 
exclude_list+=third_node_examples
valid_list = list(set(example_list) - set(exclude_list))
fourth_node_examples=random.sample(valid_list,k=9)
#Fiveth Node 
exclude_list+=fourth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
fiveth_node_examples=random.sample(valid_list,k=10)
#Sixth Node 
exclude_list+=fiveth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
sixth_node_examples=random.sample(valid_list,k=11)
#Seventh Node 
exclude_list+=sixth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
seventh_node_examples=random.sample(valid_list,k=12)
#Eighth Node 
exclude_list+=seventh_node_examples
valid_list = list(set(example_list) - set(exclude_list))
eighth_node_examples=random.sample(valid_list,k=13)
#Nineth Node 
exclude_list+=eighth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
niveth_node_examples=random.sample(valid_list,k=14)
#Tenth Node 
exclude_list+=niveth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
tenth_node_examples=random.sample(valid_list,k=6)
#Eleventh Node 
exclude_list+=tenth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
eleventh_node_examples=random.sample(valid_list,k=7)
#Twelfth Node 
exclude_list+=eleventh_node_examples
valid_list = list(set(example_list) - set(exclude_list))
twelfth_node_examples=random.sample(valid_list,k=8)
#Thirth Node 
exclude_list+=twelfth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
thirth_node_examples=random.sample(valid_list,k=10)
#th14 Node 
exclude_list+=thirth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
th14_node_examples=random.sample(valid_list,k=12)
#th15 Node 
exclude_list+=th14_node_examples
valid_list = list(set(example_list) - set(exclude_list))
th15_node_examples=random.sample(valid_list,k=17)

###### CREATE USER DATA SPLIT #######
# Assign n samples to each user
X = [[] for _ in range(15)]
y = [[] for _ in range(15)]
X[0] = iris.data[first_node_examples].tolist()
y[0] = iris.target[first_node_examples].tolist()
X[1] = iris.data[second_node_examples].tolist()
y[1] = iris.target[second_node_examples].tolist()
X[2] = iris.data[third_node_examples].tolist()
y[2] = iris.target[third_node_examples].tolist()
X[3] = iris.data[fourth_node_examples].tolist()
y[3] = iris.target[fourth_node_examples].tolist()
X[4] = iris.data[fiveth_node_examples].tolist()
y[4] = iris.target[fiveth_node_examples].tolist()
X[5] = iris.data[sixth_node_examples].tolist()
y[5] = iris.target[sixth_node_examples].tolist()
X[6] = iris.data[seventh_node_examples].tolist()
y[6] = iris.target[seventh_node_examples].tolist()
X[7] = iris.data[eighth_node_examples].tolist()
y[7] = iris.target[eighth_node_examples].tolist()
X[8] = iris.data[niveth_node_examples].tolist()
y[8] = iris.target[niveth_node_examples].tolist()
X[9] = iris.data[tenth_node_examples].tolist()
y[9] = iris.target[tenth_node_examples].tolist()
X[10] = iris.data[eleventh_node_examples].tolist()
y[10] = iris.target[eleventh_node_examples].tolist()
X[11] = iris.data[twelfth_node_examples].tolist()
y[11] = iris.target[twelfth_node_examples].tolist()
X[12] = iris.data[thirth_node_examples].tolist()
y[12] = iris.target[thirth_node_examples].tolist()
X[13] = iris.data[th14_node_examples].tolist()
y[13] = iris.target[th14_node_examples].tolist()
X[14] = iris.data[th15_node_examples].tolist()
y[14] = iris.target[th15_node_examples].tolist()
# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 1000 users
for i in trange(15, ncols=150):
    print(len(X[i]))
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

print(train_data['num_samples'])

print(test_data['num_samples'])

with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)