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

"""Tests for matching_networks_model.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import configuration
from matching_networks_model import MatchingNetworks


class MatchingNetworksModel(MatchingNetworks):
    """Subclass of MatchingNetworks without the disk I/O."""

    def build_inputs(self):
        if self.mode == "inference":
            # Inference mode doesn't read from disk, so defer to parent.
            return super(MatchingNetworks, self).build_inputs()
        else:
            # Replace disk I/O with random Tensors.
            self.support_set_sounds = tf.random_uniform(
                shape=[self.config.batch_size_s, self.config.num_mels,
                       self.config.frames, 1],
                minval=-1,
                maxval=1)
            self.test_sound = tf.random_uniform(
                shape=[self.config.batch_size_b, self.config.num_mels,
                       self.config.frames, 1],
                minval=-1,
                maxval=1)
            self.support_set_labels = tf.random_uniform(
                [self.config.batch_size_s, 1],
                minval=1,
                maxval=5,
                dtype=tf.int64)

            self.test_sound_labels = tf.random_uniform(
                [self.config.batch_size_b, 1],
                minval=1,
                maxval=5,
                dtype=tf.int64)


class MatchingNetworksModelTest(tf.test.TestCase):
    def setUp(self):
        super(MatchingNetworksModelTest, self).setUp()
        self._model_config = configuration.ModelConfig()

    def _countModelParameters(self):
        """Counts the number of parameters in the model at top level scope."""
        counter = {}
        for v in tf.all_variables():
            name = v.op.name.split("/")[0]
            num_params = v.get_shape().num_elements()
            assert num_params
            counter[name] = counter.get(name, 0) + num_params
        return counter

    def _checkModelParameters(self):
        """Verifies the number of parameters in the model."""
        param_counts = self._countModelParameters()
        expected_param_counts = {
            "InceptionV3": 21802784,
            # inception_output_size * embedding_size
            "image_embedding": 1048576,
            # vocab_size * embedding_size
            "seq_embedding": 6144000,
            # (embedding_size + num_lstm_units + 1) * 4 * num_lstm_units
            "lstm": 2099200,
            # (num_lstm_units + 1) * vocab_size
            "logits": 6156000,
            "global_step": 1,
        }
        self.assertDictEqual(expected_param_counts, param_counts)

    def _checkOutputs(self, expected_shapes, feed_dict=None):
        """Verifies that the model produces expected outputs.

        Args:
          expected_shapes: A dict mapping Tensor or Tensor name to expected output
            shape.
          feed_dict: Values of Tensors to feed into Session.run().
        """
        fetches = expected_shapes.keys()

        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            outputs = sess.run(fetches, feed_dict)

        for index, output in enumerate(outputs):
            tensor = fetches[index]
            expected = expected_shapes[tensor]
            actual = output.shape
            if expected != actual:
                self.fail("Tensor %s has shape %s (expected %s)." %
                          (tensor, actual, expected))

    def testBuildForTraining(self):
        model = MatchingNetworksModel(self._model_config, mode="train")
        model.build()

        self._checkModelParameters()

        expected_shapes = {
            # [batch_size, image_height, image_width, 3]
            model.images: (32, 299, 299, 3),
            # [batch_size, sequence_length]
            model.input_seqs: (32, 15),
            # [batch_size, sequence_length]
            model.target_seqs: (32, 15),
            # [batch_size, sequence_length]
            model.input_mask: (32, 15),
            # [batch_size, embedding_size]
            model.image_embeddings: (32, 512),
            # [batch_size, sequence_length, embedding_size]
            model.seq_embeddings: (32, 15, 512),
            # Scalar
            model.total_loss: (),
            # [batch_size * sequence_length]
            model.target_cross_entropy_losses: (480,),
            # [batch_size * sequence_length]
            model.target_cross_entropy_loss_weights: (480,),
        }
        self._checkOutputs(expected_shapes)

    def testBuildForEval(self):
        model = MatchingNetworks(self._model_config, mode="eval")
        model.build()

        self._checkModelParameters()

        expected_shapes = {
            # [batch_size, image_height, image_width, 3]
            model.images: (32, 299, 299, 3),
            # [batch_size, sequence_length]
            model.input_seqs: (32, 15),
            # [batch_size, sequence_length]
            model.target_seqs: (32, 15),
            # [batch_size, sequence_length]
            model.input_mask: (32, 15),
            # [batch_size, embedding_size]
            model.image_embeddings: (32, 512),
            # [batch_size, sequence_length, embedding_size]
            model.seq_embeddings: (32, 15, 512),
            # Scalar
            model.total_loss: (),
            # [batch_size * sequence_length]
            model.target_cross_entropy_losses: (480,),
            # [batch_size * sequence_length]
            model.target_cross_entropy_loss_weights: (480,),
        }
        self._checkOutputs(expected_shapes)

    def testBuildForInference(self):
        model = MatchingNetworks(self._model_config, mode="inference")
        model.build()

        self._checkModelParameters()

        # Test feeding an image to get the initial LSTM state.
        images_feed = np.random.rand(1, 299, 299, 3)
        feed_dict = {model.images: images_feed}
        expected_shapes = {
            # [batch_size, embedding_size]
            model.image_embeddings: (1, 512),
            # [batch_size, 2 * num_lstm_units]
            "lstm/initial_state:0": (1, 1024),
        }
        self._checkOutputs(expected_shapes, feed_dict)

        # Test feeding a batch of inputs and LSTM states to get softmax output and
        # LSTM states.
        input_feed = np.random.randint(0, 10, size=3)
        state_feed = np.random.rand(3, 1024)
        feed_dict = {"input_feed:0": input_feed, "lstm/state_feed:0": state_feed}
        expected_shapes = {
            # [batch_size, 2 * num_lstm_units]
            "lstm/state:0": (3, 1024),
            # [batch_size, vocab_size]
            "softmax:0": (3, 12000),
        }
        self._checkOutputs(expected_shapes, feed_dict)


if __name__ == "__main__":
    tf.test.main()
