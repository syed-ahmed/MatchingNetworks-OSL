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
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import configuration
import matching_networks_model
from ops import inputs as input_ops

import time
from datetime import datetime
import os

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern",
                       "/Users/luna/Desktop/osl-soundtouch/data/processed_data/train-?????-of-00005",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("model_checkpoint_file", "",
                       "Path to pretrained models g and f.")
tf.flags.DEFINE_string("train_dir", "/Users/luna/Desktop/osl-soundtouch/model/train",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_model", True,
                        "Whether to train model submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 2000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.logging.set_verbosity(tf.logging.INFO)

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.


def moving_av(total_loss):
    """
    Generates moving average for all losses

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    return loss_averages_op


def train_op_fun(total_loss, global_step):
    """Train model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    nb_ex_per_train_epoch = 1000

    num_batches_per_epoch = nb_ex_per_train_epoch / 5
    decay_steps = int(num_batches_per_epoch * 8)

    initial_learning_rate = 0.001

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = total_loss

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def main(unused_argv):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, "--train_dir is required"

    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    # model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
    training_config = configuration.TrainingConfig()

    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Build the TensorFlow graph.

    with tf.Graph().as_default():
        dataset = input_ops.process_pickles_and_augment("/Users/luna/Desktop/osl-soundtouch/data/processed_data", 0.02,
                                                        'train')
        eval_dataset = input_ops.process_pickles_and_augment("/Users/luna/Desktop/osl-soundtouch/data/processed_data", 0.02,
                                                        'validation')

        model = matching_networks_model.MatchingNetworks(
            model_config, mode="train", dataset=dataset, train_model=FLAGS.train_model)
        model.build()

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train_op_fun(model.loss, model.global_step)

        #test_acc = tf.reduce_mean(tf.to_float(model.top_k))

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())


        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.number_of_steps):

            start_time = time.time()
            batch_s_sounds, batch_s_labels = input_ops.data_iterator(dataset, model.config.batch_size_s)
            batch_s_labels = np.expand_dims(batch_s_labels, 1)
            test_sound, test_label = input_ops.data_iterator(dataset, model.config.batch_size_b)
            test_sound = np.expand_dims(test_sound[0], 0)
            test_label = np.expand_dims(test_label[0], 0).reshape((1, 1))

            # Prepare dictionnary to feed the session with
            feed_dict = {model.support_set_sounds: batch_s_sounds,
                         model.support_set_labels: batch_s_labels,
                         model.test_sound: test_sound,
                         model.test_sound_labels: test_label}

            _, train_acc, loss_val, summary = sess.run([train_op, model.train_accuracy, model.loss, summary_op], feed_dict=feed_dict)

            duration = time.time() - start_time

            assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                number_of_shot = model.config.batch_size_s / float(duration)
                format_str = ('%s: episode %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/shot) train_acc = %.4f')
                print(format_str % (datetime.now(), step, loss_val,
                                    number_of_shot, duration, train_acc))

                summary_writer.add_summary(summary, step)

            if step % 100 == 0:
                batch_s_sounds, batch_s_labels, batch_test, batch_test_label = input_ops.eval_data_iterator(dataset, eval_dataset, model.config.batch_size_s, model.config.batch_size_b)
                batch_s_labels = np.expand_dims(batch_s_labels, 1)
                batch_test = np.expand_dims(test_sound[0], 0)
                batch_test_label = np.expand_dims(test_label[0], 0).reshape((1, 1))
                # Prepare dictionnary to feed the session with
                feed_dict = {model.support_set_sounds: batch_s_sounds,
                             model.support_set_labels: batch_s_labels,
                             model.test_sound: batch_test,
                             model.test_sound_labels: batch_test_label}

                _, test_summary = sess.run([model.test_acc, model.test_summ], feed_dict=feed_dict)
                summary_writer.add_summary(test_summary, step)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary_writer.add_run_metadata(run_metadata, 'step%d' % step)

                # Save the model checkpoint periodically.
            if step % 500 == 0 or (step + 1) == FLAGS.number_of_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)



if __name__ == "__main__":
    tf.app.run()
