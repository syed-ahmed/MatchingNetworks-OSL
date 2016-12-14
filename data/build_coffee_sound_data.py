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

tf.app.flags.DEFINE_string('train_directory', "/Users/luna/Desktop/osl-soundtouch/data/segmented_data/train",
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', "/Users/luna/Desktop/osl-soundtouch/data/segmented_data/val",
                           'Validation data directory')

tf.flags.DEFINE_string("output_dir", "/Users/luna/Desktop/osl-soundtouch/data/processed_data",
                       "Output data directory.")

tf.app.flags.DEFINE_integer('train_shards', 1,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 1,
                            'Number of shards in validation TFRecord files.')

tf.flags.DEFINE_integer("num_threads", 1,
                        "Number of threads to preprocess the sound files.")

tf.flags.DEFINE_string("button_names", "power, brew, steam",
                       "Comma separated names of device buttons.")

tf.flags.DEFINE_float("silence_threshold", 0.1,
                      "Silence Threshold for trimming audio end points.")

tf.flags.DEFINE_bool("segment_audio", False,
                     "Segment raw audio into individual samples of classes.")

tf.flags.DEFINE_integer("sampling_rate", 44100,
                        "Sampling rate of the raw audio.")

tf.flags.DEFINE_string("sound_dir", "/Users/luna/Desktop/osl-soundtouch/data",
                       "Training sound directory.")

tf.flags.DEFINE_string("labels_file_dir", "/Users/luna/Desktop/osl-soundtouch/data/labels.txt",
                       "Labels file directory.")

FLAGS = tf.flags.FLAGS


def _find_files(directory, pattern):
    """
    Recursively finds all files matching the pattern.
    :param directory: directory of files
    :param pattern: file pattern to match
    :return: list of file paths
    """
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def _find_segmented_sound_files(data_dir, labels_file):
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
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    all_files_list = []

    # Leave label index 0 empty as a background class.
    label_index = 0

    # Construct the list of .wav files and labels.
    for text in unique_labels:
        labels = []
        filenames = []
        texts = []
        sound_file_path = '%s/%s/' % (data_dir, text)
        if os.path.exists(sound_file_path):
            sound_file_path += '*'
            matching_files = tf.gfile.Glob(sound_file_path)

            labels.extend([label_index] * len(matching_files))
            texts.extend([text] * len(matching_files))
            filenames.extend(matching_files)

            # Shuffle the ordering of all sound files in order to guarantee
            # random ordering of the sound files with respect to label in the
            # saved TFRecord files. Make the randomization repeatable.
            shuffled_index = range(len(filenames))
            random.seed(12345)
            random.shuffle(shuffled_index)

            filenames = [filenames[i] for i in shuffled_index]
            texts = [texts[i] for i in shuffled_index]
            labels = [labels[i] for i in shuffled_index]

            print('Found %d sound files across label number %d inside %s.' %
                  (len(filenames), label_index, sound_file_path[:-1]))

            file_label_tuple = (filenames, texts, labels)
            all_files_list.append(file_label_tuple)

            print('Finished finding files in %d of %d classes.' % (
                label_index + 1, len(unique_labels)))
        label_index += 1

    return all_files_list


def _trim_silence(audio, threshold):
    """
    Removes silence at the beginning and end of a sample

    :param audio: numpy array of audio time series
    :param threshold: rmse value of audio. deafault is 0.1
    :return: numpy array of trimmed audio

    """
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def _segment_raw_audio(directory):
    """
    Function that segments raw audio data into individual samples.
    Assumes that
        1. the raw wav file and label file has the same name.
        2. there is some silence time at the beginning and end of raw audio
        3. Silence threshold set at 0.1
    :param directory: directory where the raw files and labels from audacity are
    :return: segments samples into the following folder structure
        data
            /segmented_data
                /brew_on
                /brew_off
                /power_on
                /power_off
                /steam_on
                /steam_off
    """
    print "Segmenting raw audio..."

    # max length array of a sample
    max_length_array = []

    # list of raw audio file paths
    sound_files = _find_files(directory + "/raw_data", pattern='*.wav')

    # names of classes
    button_names = [x.strip() for x in FLAGS.button_names.split(',')]

    # loop two times
    # in first pass find the max length of a samples
    # in second pass segment the audio and write to .wav
    for m in range(0, 2):

        if m == 0:
            print "Finding the maximum time of a sound sample..."

        for i in sound_files:
            file_name = i.split("/")[-1].split(".")[0]

            # get the corresponding label file
            label_file = _find_files(directory, pattern=file_name + "*.txt")[0]

            # for each class names create directories if doesn't exist
            for j in button_names:
                try:
                    if fnmatch.fnmatch(file_name, "*" + j + "*"):
                        if not os.path.exists(os.path.join(directory, "segmented_data/" + j + "_on")):
                            os.makedirs(os.path.join(directory, "segmented_data/" + j + "_on"))
                        if not os.path.exists(os.path.join(directory, "segmented_data/" + j + "_off")):
                            os.makedirs(os.path.join(directory, "segmented_data/" + j + "_off"))

                        # load the audio
                        y, sr = librosa.load(i, sr=FLAGS.sampling_rate, mono=True)

                        # temporary array of start and end of each sample
                        onset_times = []

                        # counter variable used for appending times
                        count = 0

                        with open(label_file) as fp:
                            for line in fp:

                                # if it is first 'time' in the labels, just add it
                                # don't repeat it
                                if count == 0:
                                    onset_times.append(float(line.split("\t")[0]))
                                    count += 1
                                else:
                                    # repeat the time at -1 position
                                    onset_times.append(onset_times[-1])
                                    onset_times.append(float(line.split("\t")[0]))

                        # find the onset indices
                        onset_sample_indices = librosa.time_to_samples(onset_times, sr=sr)

                        # add the first and last sample indices of y
                        # now the onset sample indices looks like
                        # [0, start1, start1, end1, start2, end2, ...]
                        onset_sample_indices = np.hstack([0, onset_sample_indices, len(y) - 1])

                        if m == 0:
                            # start at 2 since first segment, 0 to start1, is assumed to be empty
                            for k in range(2, len(onset_sample_indices) - 1, 2):
                                # trim each segment and find their size
                                trimmed_segment = _trim_silence(y[onset_sample_indices[k]:onset_sample_indices[k + 1]],
                                                                FLAGS.silence_threshold)

                                # keep a record of the max size
                                max_length_array.append(librosa.get_duration(trimmed_segment, sr=sr))

                        else:
                            # counter for on and off samples to use in naming files
                            on_count = 0
                            off_count = 0

                            # toggle 0 and 1 for on and off pattern
                            toggle_on_off = 0

                            for k in range(2, len(onset_sample_indices) - 1, 2):

                                # segment the audio
                                sample = y[onset_sample_indices[k]:onset_sample_indices[k] + (
                                    (np.mean(max_length_array) + 0.2) * sr + 1)]

                                # put in the on/off folders
                                if toggle_on_off == 0:
                                    file_name = directory + "/segmented_data/" + j + "_on/" + j + "_on_" \
                                                + str(on_count) + ".wav"
                                    librosa.output.write_wav(file_name, sample, sr)
                                    toggle_on_off ^= 1
                                    on_count += 1
                                    print "Success: %s" % file_name
                                elif toggle_on_off == 1:
                                    file_name = directory + "/segmented_data/" + j + "_off/" + j + "_off_" \
                                                + str(off_count) + ".wav"
                                    librosa.output.write_wav(file_name, sample, sr)
                                    toggle_on_off ^= 1
                                    off_count += 1
                                    print "Success: %s" % file_name


                except:
                    print "No files matching the given classnames found."

        if m == 0:
            print "The maximum time for one sample is : %f" % (np.mean(max_length_array) + 0.2)
            print "Outputting segmented audio into .wav files now..."

        if m == 1:
            print "Audio segmentation completed."


def _process_sound(filename):
    """Process a single sound file.

    Args:
      filename: string, path to a sound file e.g., '/path/to/example.wav'.
      coder: instance of SoundCoder to provide TensorFlow sound coding utils.
    Returns:
      sound_buffer: sound samples
      total_samples: float, samples_per_second * length
      number_channels: integer, channel count
    """

    # Decode the WAV.
    sound, _ = librosa.load(filename, sr=FLAGS.sampling_rate)

    # Check that sound converted correctly
    assert len(sound.shape) == 1
    return sound


def _process_sound_files_batch(thread_index, ranges, name, filenames,
                               labels, num_shards):
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
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        # shard = thread_index * num_shards_per_batch + s
        # output_filename = '%s-%.5d-of-%.5d' % (name, num_shards, num_shards)
        output_file = os.path.join(FLAGS.output_dir, name)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        label_and_sound = {}
        sound_buffer = []
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            sound_buffer.append(_process_sound(filename))
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d sound files in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        sound_buffer = np.vstack(sound_buffer)
        label_and_sound[str(label)] = sound_buffer
        with open(output_file, 'wb') as file:
            pickle.dump(label_and_sound, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s [thread %d]: Wrote %d sound files to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d sound files to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_sound_files(name, filenames, texts, labels, num_shards):
    """Process and save list of sound files as TFRecord of Example protos.

    Args:
      filenames: list of strings; each string is a path to a sound file
      texts: list of strings; each string is human readable, e.g. 'power_on'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all sound files into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, filenames,
                labels, num_shards)
        t = threading.Thread(target=_process_sound_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d sound files in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _process_dataset(name, directory, num_shards, labels_file):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    all_files_list = _find_segmented_sound_files(directory, labels_file)
    for idx, val in enumerate(all_files_list):
        filenames, texts, labels = val
        shard_output_name = '%s-%.5d-of-%.5d.dat' % (name, idx + 1, len(all_files_list))
        _process_sound_files(shard_output_name, filenames, texts, labels, num_shards)


def main(unused_argv):
    if FLAGS.segment_audio:
        _segment_raw_audio(FLAGS.sound_dir)

    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')

    if not os.path.exists(os.path.join(FLAGS.sound_dir, "processed_data/")):
        os.makedirs(os.path.join(FLAGS.sound_dir, "processed_data/"))

    print('Saving results to %s' % FLAGS.output_dir)

    # Run it!
    _process_dataset('train', FLAGS.train_directory,
                     FLAGS.train_shards, FLAGS.labels_file_dir)
    _process_dataset('validation', FLAGS.validation_directory,
                     FLAGS.validation_shards, FLAGS.labels_file_dir)


if __name__ == "__main__":
    tf.app.run()
