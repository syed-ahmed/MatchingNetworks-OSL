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

"""Matching networks implementation based on https://arxiv.org/pdf/1606.04080v1.pdf

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ops.embedding_function import full_context_embeddings_cnn
from ops import inputs as input_ops


class MatchingNetworks(object):
    """
    Matching networks implementation based
    on https://arxiv.org/pdf/1606.04080v1.pdf
    """

    def __init__(self, config, mode, dataset, train_model=True):
        """Basic setup.

        Args:
          config: Object containing configuration parameters.
          mode: "train", "eval" or "inference".
          train_model: Whether the model's submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.dataset = dataset
        self.train_model = train_model

        # we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.support_set_sounds = None
        self.support_set_labels = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.test_sound = None
        self.test_sound_labels = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.f_embedding = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.g_embedding = None

        self.cosine_similarities = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.loss = None

        # Collection of variables from the models' submodel.
        self.g_model_variables = []
        self.f_model_variables = []

        # Function to restore the models' submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

        self.runner = None

        self.prediction = None

        self.train_accuracy = None

        self.test_acc = None

        self.test_summ = None


    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.

        Outputs:
          self.images
          self.input_seqs
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)
        """
        if self.mode == "inference":
            # In inference mode, images and inputs are fed via placeholders.
            support_set_sounds = tf.placeholder(dtype=tf.string, shape=[], name="support_set_feed")
            test_sound = tf.placeholder(dtype=tf.string, shape=[], name="test_feed")
        else:
            self.support_set_sounds = tf.placeholder(dtype=tf.float32,
                                                     shape=[self.config.batch_size_s * self.config.num_classes, 128, 64,
                                                            1])
            self.support_set_labels = tf.placeholder(dtype=tf.int64,
                                                     shape=[self.config.batch_size_s * self.config.num_classes, 1])
            self.test_sound = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size_b, 128, 64, 1])
            self.test_sound_labels = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size_b, 1])

    def build_fully_conditional_embedding_g(self):
        """Builds the fully conditional embedding g

        Inputs:
          self.sounds

        Outputs:
          self.g_embeddings
        """
        model_output = full_context_embeddings_cnn(self.support_set_sounds,
                                                   trainable=self.train_model,
                                                   is_training=self.is_training(),
                                                   scope="g_embedding_support_vectors")

        self.g_model_variables = tf.get_collection(
            tf.GraphKeys.VARIABLES, scope="g_embedding_support_vectors")

        with tf.variable_scope("fce_embedding_g") as scope:
            sound_embeddings = tf.contrib.layers.fully_connected(
                inputs=model_output,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

            sound_embeddings = tf.expand_dims(sound_embeddings, 0)
            sound_embeddings = tf.unpack(sound_embeddings)

            cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.embedding_size * 0.5,
                                              initializer=self.initializer,
                                              use_peepholes=True,
                                              state_is_tuple=True)
            # Backward direction cell
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.embedding_size * 0.5,
                                              initializer=self.initializer,
                                              use_peepholes=True,
                                              state_is_tuple=True
                                              )

            (outputs, state, _) = tf.nn.bidirectional_rnn(cell_fw,
                                                          cell_bw,
                                                          sound_embeddings,
                                                          dtype=tf.float32)

        self.g_embedding = outputs

    def build_fully_conditional_embedding_f(self):
        """Builds the fully conditional embedding f

        Inputs:
          self.sounds

        Outputs:
          self.sound_embeddings
        """
        model_output = full_context_embeddings_cnn(self.test_sound,
                                                   trainable=self.train_model,
                                                   is_training=self.is_training(),
                                                   scope="f_test_support_vector"
                                                   )

        self.f_model_variables = tf.get_collection(
            tf.GraphKeys.VARIABLES, scope="f_test_support_vector")

        with tf.variable_scope('fce_embedding_f', initializer=self.initializer) as scope:
            sound_embeddings = tf.contrib.layers.fully_connected(
                inputs=model_output,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

            # Feed the test image embeddings to set the initial LSTM state.
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.embedding_size,
                                           state_is_tuple=False,
                                           use_peepholes=True)

            zero_state = cell.zero_state(batch_size=sound_embeddings.get_shape()[0], dtype=tf.float32)

            output, initial_state = cell(sound_embeddings, zero_state)

            attention = tf.nn.softmax((tf.matmul(self.g_embedding[0], tf.transpose(output))))
            output = tf.add(model_output, output)
            read_out = tf.reduce_sum(tf.mul(attention, self.g_embedding[0]), 0, keep_dims=True)
            h_concatenated = tf.concat(1, [output, read_out])

            scope.reuse_variables()
            # Embedding shared by the input and outputs.

            for i in xrange(self.config.lstm_processing_steps):
                output, initial_state = cell(sound_embeddings, h_concatenated)
                attention = tf.nn.softmax((tf.matmul(self.g_embedding[0], tf.transpose(output))))
                output = tf.add(model_output, output)
                read_out = tf.reduce_sum(tf.mul(attention, self.g_embedding[0]), 0, keep_dims=True)
                h_concatenated = tf.concat(1, [output, read_out])

        self.f_embedding = output

    def get_cosine_similarities(self):
        with tf.variable_scope('cosine_similarity'):
            # Compute the cosine similarity between f and g embedding

            norm = tf.sqrt(tf.reduce_sum(tf.square(self.f_embedding), 1, keep_dims=True))
            normalized_f_embedding = self.f_embedding / norm
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.g_embedding), 2, keep_dims=True))
            normalized_g_embedding = tf.squeeze(self.g_embedding / norm)

            similarity = tf.matmul(
                normalized_g_embedding, normalized_f_embedding, transpose_b=True)

        self.cosine_similarities = similarity

    def build_model(self):
        """Builds the model.

        Inputs:
          self.cosine_similarities
          self.support_set_labels (training and eval only)
          self.test_sound_labels (training and eval only)


        Outputs:
          self.total_loss (training and eval only)
          self.target_cross_entropy_losses (training and eval only)
          self.target_cross_entropy_loss_weights (training and eval only)
        """

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.nn.softmax(self.cosine_similarities, dim=0)

            logits = logits * tf.contrib.slim.one_hot_encoding(tf.squeeze(self.support_set_labels), self.config.num_classes)
            logits = tf.reduce_sum(logits, 0, keep_dims=True)
            self.prediction = tf.nn.in_top_k(logits,tf.squeeze(self.test_sound_labels, squeeze_dims=[0]),1)

            #accuracy calc
            self.train_accuracy = tf.reduce_mean(tf.to_float(self.prediction))
            tf.scalar_summary('train avg accuracy', self.train_accuracy)

            self.test_acc = tf.reduce_mean(tf.to_float(self.prediction))
            self.test_summ = tf.scalar_summary('test avg accuracy', self.test_acc)


            logits = tf.expand_dims(tf.cast(tf.argmax(logits, 1), dtype=tf.float32),0)

        correct_label = tf.cast(self.test_sound_labels, dtype=tf.float32)

        tf.contrib.slim.losses.softmax_cross_entropy(logits, correct_label)
        total_loss = tf.contrib.slim.losses.get_total_loss()

        # Add summaries.
        tf.scalar_summary("losses", total_loss)
        # Add to TF collection for losses
        tf.add_to_collection('losses', total_loss)


        self.loss = total_loss

    def setup_inception_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            # Restore inception variables only.
            saver_f = tf.train.Saver(self.f_model_variables)
            saver_g = tf.train.Saver(self.g_model_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring model variables from checkpoint file %s",
                                self.config.f_checkpoint_file)
                saver_f.restore(sess, self.config.f_checkpoint_file)
                tf.logging.info("Restoring model variables from checkpoint file %s",
                                self.config.g_checkpoint_file)
                saver_g.restore(sess, self.config.g_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and val."""
        self.build_inputs()
        self.build_fully_conditional_embedding_g()
        self.build_fully_conditional_embedding_f()
        self.get_cosine_similarities()
        self.build_model()
        self.setup_inception_initializer()
        self.setup_global_step()
