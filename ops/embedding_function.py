# Copyright 2016 Syed Ahmed. All Rights Reserved.
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

"""Embedding functions for the attention kernel. f = g"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def full_context_embeddings_cnn(inputs,
                                trainable=True,
                                is_training=True,
                                weight_decay=0.00004,
                                stddev=0.1,
                                dropout_keep_prob=0.8,
                                use_batch_norm=True,
                                batch_norm_params=None,
                                add_summaries=True,
                                num_modules=4,
                                scope="FullContextEmbeddingsCNN"):
    # Only consider the model to be in training mode if it's trainable.
    is_model_training = trainable and is_training

    if use_batch_norm:
        # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "is_training": is_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "decay": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "variables_collections": {
                    "beta": None,
                    "gamma": None,
                    "moving_mean": ["moving_vars"],
                    "moving_variance": ["moving_vars"],
                }
            }
    else:
        batch_norm_params = None

    if trainable:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    with tf.variable_scope(scope, 'FullContextEmbeddingsCNN', [inputs]) as scope:

        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_regularizer=weights_regularizer,
                trainable=trainable):

            with slim.arg_scope(
                    [slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params,
                    stride=1,
                    padding='VALID'):

                with tf.variable_scope(scope, 'FullContextEmbeddingsCNN', [inputs]):
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                        stride=1, padding='VALID'):

                        for i in range(num_modules):
                            if i == 0:
                                end_point = 'Module_' + str(i)
                                with tf.variable_scope(end_point):
                                    scope_name = 'Conv2d_Module_' + str(i)
                                    net = slim.conv2d(inputs, 64, [3, 3], scope=scope_name)
                                    scope_name = 'MaxPool_Module_' + str(i)
                                    net = slim.max_pool2d(net, [2, 2], stride=2, scope=scope_name)

                                end_points[end_point] = net
                            else:
                                end_point = 'Module_' + str(i)
                                with tf.variable_scope(end_point):
                                    scope_name = 'Conv2d_Module_' + str(i)
                                    net = slim.conv2d(net, 64, [3, 3], scope=scope_name)
                                    scope_name = 'MaxPool_Module_' + str(i)
                                    net = slim.max_pool2d(net, [2, 2], stride=2, scope=scope_name)

                                end_points[end_point] = net

                        with tf.variable_scope("logits"):
                            shape = net.get_shape()
                            net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                            net = slim.dropout(
                                net,
                                keep_prob=dropout_keep_prob,
                                is_training=is_model_training,
                                scope="dropout")
                            net = slim.flatten(net, scope="flatten")

    # Add summaries.
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)

    return net
