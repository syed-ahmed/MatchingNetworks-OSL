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

import configuration
import matching_networks_model
from ops import inputs as input_ops

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
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, "--train_dir is required"

    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    #model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
    training_config = configuration.TrainingConfig()

    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        dataset = input_ops.process_pickles_and_augment("/Users/luna/Desktop/osl-soundtouch/data/processed_data", 0.02, 'train')

        model = matching_networks_model.MatchingNetworks(
            model_config, mode="train", dataset=dataset, train_model=FLAGS.train_model)
        model.build()

        # Set up the learning rate.
        learning_rate_decay_fn = None
        if FLAGS.train_model:
            learning_rate = tf.constant(training_config.train_inception_learning_rate)
        else:
            learning_rate = tf.constant(training_config.initial_learning_rate)
            if training_config.learning_rate_decay_factor > 0:
                num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                         model_config.batch_size_s)
                decay_steps = int(num_batches_per_epoch *
                                  training_config.num_epochs_per_decay)

                def _learning_rate_decay_fn(learning_rate, global_step):
                    return tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps=decay_steps,
                        decay_rate=training_config.learning_rate_decay_factor,
                        staircase=True)

                learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

    # Run training.
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver
    )


if __name__ == "__main__":
    tf.app.run()
