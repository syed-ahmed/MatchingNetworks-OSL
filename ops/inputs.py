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

"""Input ops."""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import os
import numpy as np
import random
import librosa
import time
import threading


def pitch_shift_op(input):
    # randomly distort the sound
    # generate a random pitch shift step between -4 and 4
    random.seed(12345)
    pitch_shift = random.randint(-4, 4)
    return librosa.effects.pitch_shift(input, sr=44100, n_steps=pitch_shift)


def convert_to_melspectogram_op(input):
    # Convert raw sound samples to log scaled mel spectograms
    # produces spectogram of shape [128,64]
    # @TODO: understand the effect of hop length and window size if any
    distorted_sound = librosa.feature.melspectrogram(input,
                                                     sr=44100, n_mels=128, hop_length=358)

    # Convert to log scale (dB). We'll use the peak power as reference.
    distorted_sound = librosa.logamplitude(distorted_sound, ref_power=np.max)
    distorted_sound = np.expand_dims(distorted_sound, axis=2)
    return distorted_sound


def process_pickles_and_augment(directory, augment_percent, mode):
    print "Loading and augmenting dataset..."
    dataset = []
    if mode == 'train':
        tf_record_pattern = os.path.join(directory, '%s-*' % "train")
    elif mode == 'validation':
        tf_record_pattern = os.path.join(directory, '%s-*' % "validation")
    data_files = tf.gfile.Glob(tf_record_pattern)
    for i in data_files:
        dataset.append(np.load(i))

    new_dataset = {}

    for i in dataset:
        sound_buffer = []
        b = i.values()[0]
        if mode == 'train':
            number_of_elements_to_augment = int(augment_percent * len(b))
            augment_index = random.sample(range(0, len(b)), number_of_elements_to_augment)
            for k in augment_index:
                sound_buffer.append([convert_to_melspectogram_op(pitch_shift_op(b[k]))])
        for j in b:
            sound_buffer.append([convert_to_melspectogram_op(j)])

        new_dataset[i.keys()[0]] = np.vstack(sound_buffer)
    return new_dataset


def data_iterator(dataset, batch_size):
    """ A simple data iterator """

    # shuffle labels and features
    batch_sounds = []
    batch_labels = []

    for key, value in dataset.iteritems():
        # give me five random indices between 0 and len of dataset
        idxs = random.sample(range(0, len(value)), batch_size)

        # get those images and append to batch_s_images
        for i in idxs:
            batch_sounds.append([value[i]])

            # get those labels and append to batch_s_labels
            batch_labels.append(int(key))

    shuffled_index = range(len(batch_sounds))
    random.seed(12345)
    random.shuffle(shuffled_index)

    batch_sounds = [batch_sounds[i] for i in shuffled_index]
    batch_labels = [batch_labels[i] for i in shuffled_index]
    batch_sounds = np.vstack(batch_sounds)

    return batch_sounds, batch_labels

def eval_data_iterator(dataset, eval_dataset, batch_size_s, batch_size_b):
    """ A simple data iterator """

    # shuffle labels and features
    batch_sounds = []
    batch_labels = []
    test_sound = []
    test_label = []

    for key, value in dataset.iteritems():

        if key == '0' or key == '1' or key == '2' or key == '3':
            # give me five random indices between 0 and len of dataset
            idxs = random.sample(range(0, len(value)), batch_size_s)

            # get those images and append to batch_s_images
            for i in idxs:
                batch_sounds.append([value[i]])

                # get those labels and append to batch_s_labels
                batch_labels.append(int(key))

    for key, value in eval_dataset.iteritems():
        # give me 1 random indices between 0 and len of dataset
        idxs = random.sample(range(0, len(value)), batch_size_b)

        # get those images and append to batch_s_images
        for i in idxs:
            batch_sounds.append([value[i]])

            # get those labels and append to batch_s_labels
            batch_labels.append(int(key)-1)

    for key, value in eval_dataset.iteritems():
        # give me 1 random indices between 0 and len of dataset
        idxs = random.sample(range(0, len(value)), 1)

        # get those images and append to batch_s_images
        for i in idxs:
            test_sound.append([value[i]])

            # get those labels and append to batch_s_labels
            test_label.append(int(key)-1)

    shuffled_index = range(len(batch_sounds))
    random.seed(12345)
    random.shuffle(shuffled_index)

    batch_sounds = [batch_sounds[i] for i in shuffled_index]
    batch_labels = [batch_labels[i] for i in shuffled_index]
    batch_sounds = np.vstack(batch_sounds)
    test_sound = np.vstack(test_sound)

    return batch_sounds, batch_labels, test_sound, test_label


class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """

    def __init__(self, dataset, batch_size_s, batch_size_b, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        self.batch_size_s = batch_size_s
        self.batch_size_b = batch_size_b
        self.batch_s_sounds = tf.placeholder(dtype=tf.float32, shape=[self.batch_size_s*self.num_classes, 128, 64, 1])
        self.batch_s_labels = tf.placeholder(dtype=tf.int64, shape=[self.batch_size_s*self.num_classes, 1])
        self.batch_b_sounds = tf.placeholder(dtype=tf.float32, shape=[self.batch_size_b, 128, 64, 1])
        self.batch_b_labels = tf.placeholder(dtype=tf.int64, shape=[self.batch_size_b, 1])

        # The actual queue of data. The queue contains a vector for
        # the sound features, and a scalar label.
        self.s_queue = tf.RandomShuffleQueue(shapes=[[self.batch_size_s*self.num_classes, 128, 64, 1], [self.batch_size_s*self.num_classes, 1]],
                                             dtypes=[tf.float32, tf.int64],
                                             capacity=16 * self.batch_size_s,
                                             min_after_dequeue=8 * self.batch_size_s)

        self.b_queue = tf.RandomShuffleQueue(shapes=[[self.batch_size_b, 128, 64, 1], [self.batch_size_b, 1]],
                                             dtypes=[tf.float32, tf.int64],
                                             capacity=16 * self.batch_size_b,
                                             min_after_dequeue=8 * self.batch_size_b)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.s_enqueue_op = self.s_queue.enqueue([self.batch_s_sounds, self.batch_s_labels])
        self.b_enqueue_op = self.b_queue.enqueue([self.batch_b_sounds, self.batch_b_labels])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        s_batch, s_labels = self.s_queue.dequeue_many(1)
        b_batch, b_labels = self.b_queue.dequeue_many(1)
        return tf.squeeze(s_batch, squeeze_dims=[0]), \
               tf.squeeze(s_labels, squeeze_dims=[0]), \
               tf.squeeze(b_batch, squeeze_dims=[0]), \
               tf.squeeze(b_labels, squeeze_dims=[0])

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for sounds, labels in data_iterator(self.dataset, self.batch_size_s):
            sess.run(self.s_enqueue_op, feed_dict={self.batch_s_sounds: sounds,
                                                   self.batch_s_labels: np.expand_dims(labels,1)
                                                   })

        for sounds, labels in data_iterator(self.dataset, self.batch_size_b):
            sess.run(self.b_enqueue_op, feed_dict={self.batch_b_sounds: sounds[0],
                                                   self.batch_b_labels: labels[0]
                                                   })

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
