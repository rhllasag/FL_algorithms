



from sklearn.datasets import fetch_mldata
from tqdm import trange
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

example_list=range(0,len(mnist.data))
exclude_list = []

#First Node
valid_list = list(set(example_list) - set(exclude_list))
first_node_examples=random.sample(valid_list,k=160)
#Second Node
exclude_list+=first_node_examples
valid_list = list(set(example_list) - set(exclude_list))
second_node_examples=random.sample(valid_list,k=150)
#Third Node 
exclude_list+=second_node_examples
valid_list = list(set(example_list) - set(exclude_list))
third_node_examples=random.sample(valid_list,k=140)
#Fourth Node 
exclude_list+=third_node_examples
valid_list = list(set(example_list) - set(exclude_list))
fourth_node_examples=random.sample(valid_list,k=130)
#Fiveth Node 
exclude_list+=fourth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
fiveth_node_examples=random.sample(valid_list,k=120)
#Sixth Node 
exclude_list+=fiveth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
sixth_node_examples=random.sample(valid_list,k=110)
#Seventh Node 
exclude_list+=sixth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
seventh_node_examples=random.sample(valid_list,k=100)
#Eighth Node 
exclude_list+=seventh_node_examples
valid_list = list(set(example_list) - set(exclude_list))
eighth_node_examples=random.sample(valid_list,k=90)
#Nineth Node 
exclude_list+=eighth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
niveth_node_examples=random.sample(valid_list,k=80)
#Tenth Node 
exclude_list+=niveth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
tenth_node_examples=random.sample(valid_list,k=70)
#Eleventh Node 
exclude_list+=tenth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
eleventh_node_examples=random.sample(valid_list,k=110)
#Twelfth Node 
exclude_list+=eleventh_node_examples
valid_list = list(set(example_list) - set(exclude_list))
twelfth_node_examples=random.sample(valid_list,k=150)
#Thirth Node 
exclude_list+=twelfth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
thirth_node_examples=random.sample(valid_list,k=150)
#th14 Node 
exclude_list+=thirth_node_examples
valid_list = list(set(example_list) - set(exclude_list))
th14_node_examples=random.sample(valid_list,k=150)
#th15 Node 
exclude_list+=th14_node_examples
valid_list = list(set(example_list) - set(exclude_list))
th15_node_examples=random.sample(valid_list,k=150)

###### CREATE USER DATA SPLIT #######
# Assign n samples to each user
X = [[] for _ in range(15)]
y = [[] for _ in range(15)]

X[0] = mnist.data[first_node_examples].tolist()
y[0] = mnist.target[first_node_examples].tolist()
X[1] = mnist.data[second_node_examples].tolist()
y[1] = mnist.target[second_node_examples].tolist()
X[2] = mnist.data[third_node_examples].tolist()
y[2] = mnist.target[third_node_examples].tolist()
X[3] = mnist.data[fourth_node_examples].tolist()
y[3] = mnist.target[fourth_node_examples].tolist()
X[4] = mnist.data[fiveth_node_examples].tolist()
y[4] = mnist.target[fiveth_node_examples].tolist()
X[5] = mnist.data[sixth_node_examples].tolist()
y[5] = mnist.target[sixth_node_examples].tolist()
X[6] = mnist.data[seventh_node_examples].tolist()
y[6] = mnist.target[seventh_node_examples].tolist()
X[7] = mnist.data[eighth_node_examples].tolist()
y[7] = mnist.target[eighth_node_examples].tolist()
X[8] = mnist.data[niveth_node_examples].tolist()
y[8] = mnist.target[niveth_node_examples].tolist()
X[9] = mnist.data[tenth_node_examples].tolist()
y[9] = mnist.target[tenth_node_examples].tolist()
X[10] = mnist.data[eleventh_node_examples].tolist()
y[10] = mnist.target[eleventh_node_examples].tolist()
X[11] = mnist.data[twelfth_node_examples].tolist()
y[11] = mnist.target[twelfth_node_examples].tolist()
X[12] = mnist.data[thirth_node_examples].tolist()
y[12] = mnist.target[thirth_node_examples].tolist()
X[13] = mnist.data[th14_node_examples].tolist()
y[13] = mnist.target[th14_node_examples].tolist()
X[14] = mnist.data[th15_node_examples].tolist()
y[14] = mnist.target[th15_node_examples].tolist()


# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 1000 users
for i in trange(15, ncols=120):
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