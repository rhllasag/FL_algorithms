import json
import numpy as np
import tensorflow as tf
from tqdm import trange
import keras
import keras.backend as K
from tensorflow.contrib import rnn

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size, process_grad

def pre_process_x(raw_x_batch):
    data_matrix = np.array(raw_x_batch)
    num_elements = data_matrix.shape[0] # Can be < 50 in test splits
    list_data=[]
    size_seq=50
    if size_seq>num_elements:
        list_data.append(data_matrix)
    else:
        for start, stop in zip(range(0, num_elements-size_seq), range(size_seq, num_elements)):
            list_data.append(data_matrix[start:stop, :])
    return list_data

def process_x(raw_x_batch):
    data_matrix = np.array(raw_x_batch)
    num_elements = data_matrix.shape[0]
    new_data=pre_process_x(raw_x_batch)
    new_data=np.concatenate(new_data).astype(np.float32)
    size_seq=50
    dim=int(new_data.shape[0]/size_seq)
    if dim>0:
        aux=np.reshape(new_data,(dim,50,25))
    else:
        aux=np.reshape(new_data,(1,num_elements,25))
    return (aux)

def process_y(raw_y_batch):
    data_matrix = np.array(raw_y_batch)
    num_elements = data_matrix.shape[0]
    size_seq=50    
    if num_elements>=50:
        return list(data_matrix[size_seq:num_elements]/1.0)
    else:
        return list(data_matrix[0:num_elements]/1.0)
    return 0

class Model(object):

    def __init__(self, seq_len, size_seq, n_hidden, optimizer, seed):
        #params
        self.size_seq=size_seq
        self.seq_len = seq_len
        self.n_hidden = n_hidden
        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, optimizer):
        features = tf.placeholder(tf.float32, [None,None, self.seq_len], name='features') # Seq 24
        labels = tf.placeholder(tf.float32, [None,], name='labels')

        stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])#two layers LSTM  # n_hidden 100
        
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, features, dtype=tf.float32)
        
        pred = tf.squeeze(tf.layers.dense(inputs=outputs[:,-1,:], units=1)) # squeeze remove 1 of input or an specific axis
        
        loss = tf.losses.mean_squared_error(labels, pred)
        # MEAN ABSOLUTE ERROR implementation
        abs_difference=tf.abs((labels)-(pred))
        abs_difference=tf.divide(abs_difference,tf.cast(tf.count_nonzero(abs_difference),tf.float32))
        mean_absolute_error=tf.reduce_sum(abs_difference)
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        
        return features, labels, train_op, grads, mean_absolute_error, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        
        grads = np.zeros(model_len)
        num_samples = len(data['y'])
        processed_samples = 0

        input_data = process_x(data['x'])
        target_data = process_y(data['y'])
        with self.graph.as_default():
            model_grads = self.sess.run(self.grads, 
                feed_dict={self.features: input_data, self.labels: target_data})
            grads = process_grad(model_grads)
        processed_samples = num_samples

        return processed_samples, grads
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            comp: number of FLOPs computed while training given data
            update: list of np.ndarray weights, with each weight array
        corresponding to a variable in the resulting graph
        '''
        
        for _ in trange(num_epochs, desc='Epoch: ', leave=False):
            for X,y in batch_data(data, batch_size):
                input_data = process_x(X)
                target_data = process_y(y)
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        '''Solves local optimization problem'''

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            input_data = process_x(X)
            target_data = process_y(y)
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = 0
        return soln, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        x_vecs = process_x(data['x'])
        labels = process_y(data['y'])
        with self.graph.as_default():
            mae, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels})
        return mae, loss
    
    def close(self):
        self.sess.close()
