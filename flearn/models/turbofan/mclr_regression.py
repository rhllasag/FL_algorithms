import numpy as np
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad

def reshape(x,_batch_size):
    return tf.reshape(x, tf.stack([_batch_size, -1]))

class Model(object):
    '''
    Assumes that are analized 25 time-series data
    '''
    
    def __init__(self, num_classes, optimizer, seed=1):

        # params
        self.num_classes = num_classes

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
        """Model function for Logistic Regression."""
        features = tf.placeholder(tf.float32, shape=[None,25], name='features')
        labels = tf.placeholder(tf.float32, shape=[1,], name='labels')
        hidden_layer1 = tf.layers.dense(inputs=features, units=40, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),activation=tf.nn.relu,use_bias=True)
        pred = tf.layers.dense(inputs=hidden_layer1, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001), units=1)
        label=reshape(labels,1)
        loss = tf.losses.mean_squared_error(label/361.0,pred)
        # MEAN ABSOLUTE ERROR implementation
        abs_difference=tf.abs((labels/361.0)-(pred))
        abs_difference=tf.divide(abs_difference,tf.cast(tf.count_nonzero(abs_difference),tf.float32))
        mean_absolute_error=tf.reduce_sum(abs_difference)

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

        global_grads = np.zeros(model_len)
        grads = np.zeros(model_len)
        num_samples = len(data['y'])
        batch_size=1
        for X, y in batch_data(data, batch_size):
            with self.graph.as_default():
                model_grads = self.sess.run(self.grads,feed_dict={self.features: X, self.labels: y})
                grads = process_grad(model_grads)
                global_grads = np.add(global_grads,grads)
        global_grads=global_grads/(num_samples*1.0)
        return num_samples, global_grads
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        batch_size=1
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        '''Solves local optimization problem'''

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = 0
        return soln, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        tot_correct=0
        loss=0
        for X, y in batch_data(data, 1): # Batch size
            with self.graph.as_default():
                tot_correct_batch, loss_batch = self.sess.run([self.eval_metric_ops, self.loss], 
                feed_dict={self.features: X, self.labels: y})
                tot_correct=tot_correct+tot_correct_batch
                loss=loss+loss_batch
        return tot_correct, loss
    
    def close(self):
        self.sess.close()
