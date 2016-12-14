# import tensorflow as tf
# import librosa
# from tensorflow.contrib import ffmpeg
# import tensorflow.contrib.slim as slim
# import os
#
# input_file_pattern = "/Users/luna/Desktop/osl-soundtouch/data/processed_data/train-?????-of-00005"
# tf_record_pattern = os.path.join("/Users/luna/Desktop/osl-soundtouch/data/processed_data", '%s-*' % "train")
# data_files = tf.gfile.Glob(tf_record_pattern)
#
#
#
#
# min_queue_examples = 90 * 2
# capacity = min_queue_examples + 100 * 5
# values_queue = tf.RandomShuffleQueue(
#     capacity=capacity,
#     min_after_dequeue=min_queue_examples,
#     dtypes=[tf.string, tf.string])
#
#
# num_readers = len(data_files)
# p_reader = slim.parallel_reader.ParallelReader(
#     tf.TFRecordReader,
#     values_queue,
#     num_readers=num_readers)
#
# data_files = slim.parallel_reader.get_data_files(data_files)
# filename_queue = tf.train.string_input_producer(data_files)
# key, value = p_reader.read(filename_queue)
#
# enqueue_ops = []
# enqueue_ops.append(values_queue.enqueue([value]))
#
# tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
#   values_queue, enqueue_ops))
#
# sound_and_labels = []
# for thread_id in range(5):
#     serialized_sequence_example = values_queue.dequeue()
#     sound, label, text = parse_example_proto(serialized_sequence_example)
#     sound_and_labels.append([sound, label,text])
#
# # Batch inputs.
# queue_capacity = (2 * 4 *
#                   5)
#
# sounds, labels,texts = tf.train.batch_join(
#     sound_and_labels,
#     batch_size=4,
#     capacity=queue_capacity,
#     dynamic_pad=True,
#     name="batch_and_pad")
#
#
# with tf.Session() as sess:
#
#
#     tf.train.start_queue_runners(sess=sess)
#
#     # grab examples back.
#     # first example from file
#
#     key, val = sess.run([key, value])
#     key1, val1 = sess.run([key, value])
#     print "end"
# # second example from file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.data import test_utils

import os


shared_queue = tf.FIFOQueue(capacity=256, dtypes=[tf.string])
tf_record_pattern = os.path.join("/Users/luna/Desktop/osl-soundtouch/data/processed_data", '%s-*' % "train")
data_files = tf.gfile.Glob(tf_record_pattern)

filename_queue1 = tf.train.string_input_producer(
    [data_files[0]], shuffle=False, capacity=1)
filename_queue2 = tf.train.string_input_producer(
    [data_files[1]], shuffle=False, capacity=1)
filename_queue3 = tf.train.string_input_producer(
    [data_files[2]], shuffle=False, capacity=1)

from ops import sound_processing
def read_my_file_format(filename_queue):
  reader = tf.TFRecordReader()
  key, record_string = reader.read(filename_queue)
  example, label = parse_example_proto(record_string)
  example = sound_processing.decode_wav(example)
  return example, label


# def input_pipeline(filenames, batch_size, num_epochs=None):
#   filename_queue = tf.train.string_input_producer(
#       filenames, num_epochs=num_epochs, shuffle=True)
#   example, label = read_my_file_format(filename_queue)
#
#


enqueue_ops = []
images = []
labels = []
image1, label1 = read_my_file_format(filename_queue1)
images.append(image1)
labels.append(label1)
image2, label2 = read_my_file_format(filename_queue2)
images.append(image2)
labels.append(label2)
# image3, label3 = read_my_file_format(filename_queue3)
# images.append(image3)
# labels.append(label3)
# all_images = tf.pack(images)
# all_labels = tf.pack(labels)
#
# min_after_dequeue = 256
# capacity = min_after_dequeue + 3 * 5
# example_batch, label_batch = tf.train.shuffle_batch(
#   [all_images, all_labels], batch_size=5, capacity=capacity, enqueue_many=True, shapes=[[22615,1],[1]],
#   min_after_dequeue=min_after_dequeue)

#enqueue_ops.append(shared_queue.enqueue(values))


# tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
#     shared_queue, enqueue_ops))
#
# example_serialized = shared_queue.dequeue()
# images, labels = parse_example_proto(example_serialized)
#
# count0 = 0
# count1 = 0
# count2 = 0
#
# num_reads = 30




sess = tf.Session()

# Required. See below for explanation
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
example_batch1, label_batch1 = sess.run([images, labels])

print ("end")