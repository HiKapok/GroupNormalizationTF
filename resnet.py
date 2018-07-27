# Copyright 2018 Changan Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ResNetGN(object):
    def __init__(self, depth=50, data_format='channels_last', gn_epsilon=1e-5):
        super(ResNetGN, self).__init__()
        self._depth = depth
        self._group = 32
        self._data_format = data_format
        self._gn_epsilon = gn_epsilon
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer
        self._block_settings = {
            50:  (3, 4, 6,  3),
            101: (3, 4, 23, 3),
        }
    # BGR, [-128, 128]
    def get_model(self, inputs, training=False):
        with tf.variable_scope('resnet', [inputs]):
            input_depth = [128, 256, 512, 1024] # the input depth of the the first block is dummy input
            num_units = self._block_settings[self._depth]

            with tf.variable_scope('block_0', [inputs]) as sc:
                if self._data_format == 'channels_first':
                    inputs = tf.pad(inputs, paddings = [[0, 0], [0, 0], [3, 3], [3, 3]], name='padding_conv1')
                else:
                    inputs = tf.pad(inputs, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]], name='padding_conv1')
                inputs = self.conv_gn_relu(inputs, input_depth[0] // 2, (7, 7), (2, 2), 'conv_1', training, reuse=None, padding='valid')
                if self._data_format == 'channels_first':
                    inputs = tf.pad(inputs, paddings = [[0, 0], [0, 0], [1, 1], [1, 1]], constant_values=float('-Inf'), name='padding_pool')
                else:
                    inputs = tf.pad(inputs, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=float('-Inf'), name='padding_pool')
                inputs = tf.layers.max_pooling2d(inputs, [3, 3], [2, 2], padding='valid', data_format=self._data_format, name='pool_1')

            is_root = True
            for ind, num_unit in enumerate(num_units):
                with tf.variable_scope('block_{}'.format(ind+1), [inputs]):
                    need_reduce = True
                    for unit_index in range(1, num_unit+1):
                        inputs = self.bottleneck_block(inputs, input_depth[ind], 'conv_{}'.format(unit_index), training, need_reduce=need_reduce, is_root=is_root)
                        need_reduce = False
                        is_root = False

            return inputs

    def group_normalization(self, inputs, training, group, scope=None):
        with tf.variable_scope(scope, 'group_normalization', [inputs], reuse=tf.AUTO_REUSE):
            if self._data_format == 'channels_last':
                _, H, W, C = inputs.get_shape().as_list()
                gamma = tf.get_variable('scale', shape=[C], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=training)
                beta = tf.get_variable('bias', shape=[C], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=training)
                inputs = tf.reshape(inputs, [-1, H, W, group, C // group], name='unpack')
                mean, var = tf.nn.moments(inputs, [1, 2, 4], keep_dims=True)
                inputs = (inputs - mean) / tf.sqrt(var + self._gn_epsilon)
                inputs = tf.reshape(inputs, [-1, H, W, C], name='pack')
                gamma = tf.reshape(gamma, [1, 1, 1, C], name='reshape_gamma')
                beta = tf.reshape(beta, [1, 1, 1, C], name='reshape_beta')
                return inputs * gamma + beta
            else:
                _, C, H, W = inputs.get_shape().as_list()
                gamma = tf.get_variable('scale', shape=[C], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=training)
                beta = tf.get_variable('bias', shape=[C], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=training)
                inputs = tf.reshape(inputs, [-1, group, C // group, H, W], name='unpack')
                mean, var = tf.nn.moments(inputs, [2, 3, 4], keep_dims=True)
                inputs = (inputs - mean) / tf.sqrt(var + self._gn_epsilon)
                inputs = tf.reshape(inputs, [-1, C, H, W], name='pack')
                gamma = tf.reshape(gamma, [1, C, 1, 1], name='reshape_gamma')
                beta = tf.reshape(beta, [1, C, 1, 1], name='reshape_beta')
                return inputs * gamma + beta

    def bottleneck_block(self, inputs, filters, scope, training, need_reduce=True, is_root=False, reuse=None):
        with tf.variable_scope(scope, 'bottleneck_block', [inputs]):
            strides = 1 if (not need_reduce) or is_root else 2
            shortcut = self.conv_gn(inputs, filters * 2, (1, 1), (strides, strides), 'shortcut', training, padding='valid', reuse=reuse) if need_reduce else inputs

            inputs = self.conv_gn_relu(inputs, filters // 2, (1, 1), (1, 1), 'reduce', training, reuse=reuse)
            if self._data_format == 'channels_first':
                inputs = tf.pad(inputs, paddings = [[0, 0], [0, 0], [1, 1], [1, 1]], name='padding_conv_3x3')
            else:
                inputs = tf.pad(inputs, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]], name='padding_conv_3x3')
            inputs = self.conv_gn_relu(inputs, filters // 2, (3, 3), (strides, strides), 'block_3x3', training, padding='valid', reuse=reuse)
            inputs = self.conv_gn(inputs, filters * 2, (1, 1), (1, 1), 'increase', training, reuse=reuse)

            return tf.nn.relu(inputs + shortcut)

    def conv_relu(self, inputs, filters, kernel_size, strides, scope, padding='same', reuse=None):
        with tf.variable_scope(scope, 'conv_relu', [inputs]):
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=True, padding=padding,
                                data_format=self._data_format, activation=tf.nn.relu,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=tf.zeros_initializer(), reuse=reuse)
            return inputs
    def conv_gn_relu(self, inputs, filters, kernel_size, strides, scope, training, padding='same', reuse=None):
        with tf.variable_scope(scope, 'conv_gn_relu', [inputs]):
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=False, padding=padding,
                                data_format=self._data_format, activation=None,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=None, reuse=reuse)

            inputs = self.group_normalization(inputs, training, self._group, scope='gn')
            return tf.nn.relu(inputs)
    def gn_relu(self, inputs, scope, training, reuse=None):
        with tf.variable_scope(scope, 'gn_relu', [inputs]):
            inputs = self.group_normalization(inputs, training, self._group, scope='gn')
            return tf.nn.relu(inputs)
    def conv_gn(self, inputs, filters, kernel_size, strides, scope, training, padding='same', reuse=None):
        with tf.variable_scope(scope, 'conv_gn', [inputs]):
            inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                                name='conv2d', use_bias=False, padding=padding,
                                data_format=self._data_format, activation=None,
                                kernel_initializer=self._conv_initializer(),
                                bias_initializer=None, reuse=reuse)
            inputs = self.group_normalization(inputs, training, self._group, scope='gn')
            return inputs

'''simple testting
'''
import numpy as np

tf.reset_default_graph()

input_image = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_placeholder')
outputs = ResNetGN().get_model(input_image)

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    saver.restore(sess, "./resnet50/resnet50.ckpt")

    # predict = sess.run(outputs, feed_dict = {input_image : np.expand_dims(np.concatenate([np.ones((1, 224,224))*0.2, np.ones((1, 224,224))*0.4, np.ones((1, 224,224))*0.6], axis=0), axis=0)})
    # predict = np.mean(predict, axis=(2, 3))
    predict = sess.run(outputs, feed_dict = {input_image : np.expand_dims(np.concatenate([np.ones((224,224,1))*0.2, np.ones((224,224,1))*0.4, np.ones((224,224,1))*0.6], axis=2), axis=0)})
    predict = np.mean(predict, axis=(1, 2))

    print(predict.shape)
    print(predict.tolist(), np.argmax(predict))
