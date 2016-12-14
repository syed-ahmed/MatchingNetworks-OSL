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

    while True:
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

        yield batch_sounds, batch_labels


class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_sounds = tf.placeholder(dtype=tf.float32, shape=[128,64,1])
        self.batch_labels = tf.placeholder(dtype=tf.int64, shape=[1])

        # The actual queue of data. The queue contains a vector for
        # the sound features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(shapes=[[128, 64, 1], [1]],
                                           dtypes=[tf.float32, tf.int64],
                                           capacity=16 * self.batch_size,
                                           min_after_dequeue=8 * self.batch_size,)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue([self.batch_sounds, self.batch_labels])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        sound_batch, labels_batch = self.queue.dequeue_many(self.batch_size)
        return sound_batch, labels_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for sounds, labels in data_iterator(self.dataset, self.batch_size):
            sess.run(self.enqueue_op, feed_dict={self.batch_sounds: sounds,
                                                 self.batch_labels: labels
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

