import numpy as np
import tensorflow as tf


class DQN:

    def __init__(self, session, input_size, output_size, name='main'):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=10, l_rate=1e-1):
        with tf.variable_scope(self.net_name):
            self._x=tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            X_img = tf.reshape(self._x, [-1, 528, 838, 1])
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding='SAME')
            L3_flat = tf.reshape(L3, [-1, 128 * 66 * 105])





            w1 = tf.get_variable('w1', shape=[128 * 66 * 105, h_size], initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(L3_flat, w1))

            w2= tf.get_variable('w2', shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

            self._qpred = tf.matmul(layer1,w2)

        self._y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._y - self._qpred))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._qpred, feed_dict={self._x:x})
    def update(self, x_stack, y_stack):
        return self.session.run(self._loss, feed_dict={self._x:x_stack, self._y:y_stack})
