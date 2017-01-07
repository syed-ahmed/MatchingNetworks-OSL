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
"""
Modified version of build_image_data.py from tensorflow/models/inception/inception/data
Converts sound data to TFRecords file format with Example protos.
The sound data set is expected to reside in tif files located in the
following directory structure.
  ...
  segmented_data/power_on/01.wav
  segmented_data/power_off/02.wav
  ...

This TensorFlow script converts the data into
a sharded data set consisting of TFRecord files
  output_directory/data-00000-of-0124
  output_directory/data-00001-of-0124
  ...
  output_directory/data-00127-of-0124

Each record within the TFRecord file is a serialized Example proto.
"""

from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import os.path
import random
import sys
import threading
import librosa
import numpy as np
import fnmatch
import tensorflow as tf
from tensorflow.contrib import ffmpeg
import pickle

tf.app.flags.DEFINE_string('train_directory', "/Users/luna/workspace/MatchingNetworks-OSL/data/omniglot/raw-data/train",
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory',
                           "/Users/luna/workspace/MatchingNetworks-OSL/data/omniglot/raw-data/val",
                           'Validation data directory')

tf.flags.DEFINE_string("output_dir", "/Users/luna/workspace/MatchingNetworks-OSL/data/omniglot/processed-data",
                       "Output data directory.")

tf.flags.DEFINE_string("data_dir", "/Users/luna/workspace/MatchingNetworks-OSL/data/omniglot",
                       "Omniglot data directory.")

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')

FLAGS = tf.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    colorspace = 'GRAY'
    channels = 1
    image_format = 'PNG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(text),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes PNG data
        self._png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._png_data)

    def decode_png(self, image_data):
        image = self._sess.run(self._decode_png,
                               feed_dict={self._png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 1
        return image


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_png(image_data)

    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 1

    return image_data, height, width


def _find_image_files(data_dir, name):
    """Build a list of all sound files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of sound files.

        Assumes that the sound data set resides in .wav files located in
        the following directory structure.

          data_dir/power_on/01.wav
          data_dir/brew_on/01.wav

        where 'power_on' is the label associated with these sound files.

      labels_file: string, path to the labels file.

        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          power_on
          power_off
          brew_on
          brew_off
          steam_on
          steam_off
        where each line corresponds to a label. We map each label contained in
        the file to one hot encoding starting with 0 corresponding to the
        label contained in the first line.

    Returns:
        a tuple of
          filenames: list of strings; each string is a path to a sound file.
          texts: list of strings; each string is the class, e.g. 'power_on'
          labels: list of one hot encoding; each encoding identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)

    all_files_list = []
    matching_files = []
    found_image = False
    label_index = 0
    labels_file_output = []

    for root, dirnames, filenames in os.walk(data_dir):
        for filename in fnmatch.filter(filenames, '*.png'):
            matching_files.append(os.path.join(root, filename))
            found_image = True

        if found_image:
            result_filenames = []
            labels = []
            texts = []

            text = root.split('/')[-2] + '_' + root.split('/')[-1]
            labels.extend([label_index] * len(matching_files))
            texts.extend([text] * len(matching_files))
            result_filenames.extend(matching_files)
            labels_file_output.append(text)

            shuffled_index = range(len(result_filenames))
            random.shuffle(shuffled_index)
            result_filenames = [result_filenames[i] for i in shuffled_index]

            print('Found %d image files across label number %d inside %s.' %
                  (len(filenames), label_index, text))

            file_label_tuple = (result_filenames, texts, labels)
            all_files_list.append(file_label_tuple)
            label_index += 1
            found_image = False
            matching_files = []

    print('Writing labels file to %s' % FLAGS.output_dir)

    if name == 'train':
        label_filename = FLAGS.output_dir + '/train_labels.txt'
    else:
        label_filename = FLAGS.output_dir + '/validation_labels.txt'
    with open(label_filename, 'wt') as file:
        file.write('\n'.join(labels_file_output))

    print('Finished finding files for %d classes.' % len(all_files_list))

    return all_files_list


def _process_image_files_batch(coder, thread_index, ranges, name, files):
    """Processes and saves list of sound files as TFRecord in 1 thread.

    Args:
      coder: instance of SoundCoder to provide TensorFlow sound coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to a sound file
      texts: list of strings; each string is human readable, e.g. 'power_on'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """

    counter = 0
    shard_counter = 0
    for idx in range(ranges[thread_index][0], ranges[thread_index][1]):

        filenames, texts, labels = files[idx]

        assert len(filenames) == len(texts)
        assert len(filenames) == len(labels)
        shard_output_name = '%s-%.5d-of-%.5d' % (name, idx + 1, len(files))
        output_file = os.path.join(FLAGS.output_dir, shard_output_name)
        writer = tf.python_io.TFRecordWriter(output_file)

        for i in xrange(len(filenames)):
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width)
            writer.write(example.SerializeToString())
            counter += 1

            if counter % 10:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, len(filenames)))

                sys.stdout.flush()

        counter = 0
        writer.close()

        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, len(filenames), output_file))
        sys.stdout.flush()
        shard_counter += 1

    print('%s [thread %d]: Wrote %d shards.' %
          (datetime.now(), thread_index, shard_counter))
    sys.stdout.flush()
    shard_counter = 0


def _process_image_files(files, name):
    """Process and save list of sound files as TFRecord of Example protos.

    Args:
      filenames: list of strings; each string is a path to a sound file
      texts: list of strings; each string is human readable, e.g. 'power_on'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Break all sound files into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(files), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, files)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d classes in data set.' %
          (datetime.now(), len(files)))
    sys.stdout.flush()


def _process_dataset(name, directory):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
    """
    all_files_list = _find_image_files(directory, name)
    _process_image_files(all_files_list, name)


def main(unused_argv):
    if not os.path.exists(os.path.join(FLAGS.data_dir, "processed-data/")):
        os.makedirs(os.path.join(FLAGS.data_dir, "processed-data/"))

    print('Saving results to %s' % FLAGS.output_dir)

    # Run it!
    _process_dataset('train', FLAGS.train_directory)
    _process_dataset('validation', FLAGS.validation_directory)


if __name__ == "__main__":
    tf.app.run()
