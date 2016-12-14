"""Tests for tensorflow_models.im2txt.ops.image_embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ops.embedding_function import full_context_embeddings_cnn


class FullContextEmbeddingsCNNTest(tf.test.TestCase):
    def setUp(self):
        super(FullContextEmbeddingsCNNTest, self).setUp()
        batch_size = 1
        height = 128
        width = 64
        num_channels = 1
        self._images = tf.placeholder(tf.float32,
                                      [batch_size, height, width, num_channels])
        self._batch_size = batch_size

    def _countInceptionParameters(self):
        """Counts the number of parameters in the model at top scope."""
        counter = {}
        for v in tf.all_variables():
            name_tokens = v.op.name.split("/")
            if name_tokens[0] == "FullContextEmbeddingsCNN":
                name = "FullContextEmbeddingsCNN/" + name_tokens[1]
                num_params = v.get_shape().num_elements()
                assert num_params
                counter[name] = counter.get(name, 0) + num_params
        return counter

    def _verifyParameterCounts(self):
        """Verifies the number of parameters in the inception model."""
        param_counts = self._countInceptionParameters()
        expected_param_counts = {
            "FullContextEmbeddingsCNN/Module_0": 768,
            "FullContextEmbeddingsCNN/Module_1": 37056,
            "FullContextEmbeddingsCNN/Module_2": 37056,
            "FullContextEmbeddingsCNN/Module_3": 37056,
        }
        self.assertDictEqual(expected_param_counts, param_counts)

    def _assertCollectionSize(self, expected_size, collection):
        actual_size = len(tf.get_collection(collection))
        if expected_size != actual_size:
            self.fail("Found %d items in collection %s (expected %d)." %
                      (actual_size, collection, expected_size))

    def testTrainableTrueIsTrainingTrue(self):
        embeddings = full_context_embeddings_cnn(
            self._images, trainable=True, is_training=True)
        self.assertEqual([self._batch_size, 64], embeddings.get_shape().as_list())

        self._verifyParameterCounts()

        self._assertCollectionSize(16, tf.GraphKeys.VARIABLES)
        self._assertCollectionSize(8, tf.GraphKeys.TRAINABLE_VARIABLES)
        self._assertCollectionSize(8, tf.GraphKeys.UPDATE_OPS)
        self._assertCollectionSize(4, tf.GraphKeys.REGULARIZATION_LOSSES)
        self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
        self._assertCollectionSize(4, tf.GraphKeys.SUMMARIES)

    def testTrainableTrueIsTrainingFalse(self):
        embeddings = full_context_embeddings_cnn(
            self._images, trainable=True, is_training=False)
        self.assertEqual([self._batch_size, 64], embeddings.get_shape().as_list())

        self._verifyParameterCounts()
        self._assertCollectionSize(16, tf.GraphKeys.VARIABLES)
        self._assertCollectionSize(8, tf.GraphKeys.TRAINABLE_VARIABLES)
        self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
        self._assertCollectionSize(4, tf.GraphKeys.REGULARIZATION_LOSSES)
        self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
        self._assertCollectionSize(4, tf.GraphKeys.SUMMARIES)

    def testTrainableFalseIsTrainingTrue(self):
        embeddings = full_context_embeddings_cnn(
            self._images, trainable=False, is_training=True)
        self.assertEqual([self._batch_size, 64], embeddings.get_shape().as_list())

        self._verifyParameterCounts()
        self._assertCollectionSize(16, tf.GraphKeys.VARIABLES)
        self._assertCollectionSize(0, tf.GraphKeys.TRAINABLE_VARIABLES)
        self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
        self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
        self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
        self._assertCollectionSize(4, tf.GraphKeys.SUMMARIES)

    def testTrainableFalseIsTrainingFalse(self):
        embeddings = full_context_embeddings_cnn(
            self._images, trainable=False, is_training=False)
        self.assertEqual([self._batch_size, 64], embeddings.get_shape().as_list())

        self._verifyParameterCounts()
        self._assertCollectionSize(16, tf.GraphKeys.VARIABLES)
        self._assertCollectionSize(0, tf.GraphKeys.TRAINABLE_VARIABLES)
        self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
        self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
        self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
        self._assertCollectionSize(4, tf.GraphKeys.SUMMARIES)


if __name__ == "__main__":
    tf.test.main()
