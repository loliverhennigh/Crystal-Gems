
import os
import numpy as np
import tensorflow as tf
from glob import glob as glb
import mineral


FLAGS = tf.app.flags.FLAGS

def read_data_mineral(filename_queue):
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'crystal_system':tf.FixedLenFeature([],tf.string),
      'fracture':tf.FixedLenFeature([],tf.string),
      'groups':tf.FixedLenFeature([],tf.string),
      'rock_type':tf.FixedLenFeature([],tf.string),
      'image':tf.FixedLenFeature([],tf.string),
    }) 
  crystal_system = tf.decode_raw(features['crystal_system'], tf.uint8)
  fracture       = tf.decode_raw(features['fracture'], tf.uint8)
  groups         = tf.decode_raw(features['groups'], tf.uint8)
  rock_type      = tf.decode_raw(features['rock_type'], tf.uint8)
  image          = tf.decode_raw(features['image'], tf.uint8)
  crystal_system = tf.reshape(crystal_system, [len(mineral.CRYSTAL_SYSTEM)])
  fracture = tf.reshape(fracture, [len(mineral.FRACTURES)])
  groups = tf.reshape(groups, [len(mineral.GROUPS)])
  rock_type = tf.reshape(rock_type, [len(mineral.ROCK_TYPES)])
  image = tf.reshape(image, [1, 110, 110, 3])
  crystal_system = tf.to_float(crystal_system)
  fracture = tf.to_float(fracture)
  groups = tf.to_float(groups)
  rock_type = tf.to_float(rock_type)
  image = tf.to_float(image)
  image = tf.image.resize_nearest_neighbor(image, [112, 112])
  image = tf.reshape(image, [112, 112, 3])
  return image, crystal_system, fracture, groups, rock_type

def read_data_cifar10(filename_queue):
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32 
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  record_bytes = label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)
  record_bytes = tf.decode_raw(value, tf.uint8)
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  return result

def _generate_image_label_batch_mineral(image, crystal_system, fracture, groups, rock_type, batch_size, shuffle=True):
  num_preprocess_threads = 1
  #Create a queue that shuffles the examples, and then
  images, crystal_systems, fractures, groupss, rock_types = tf.train.shuffle_batch(
    [image, crystal_system, fracture, groups, rock_type],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=1000 + 3 * batch_size,
    min_after_dequeue=1000)
  return images, crystal_systems, fractures, groupss, rock_types


def _generate_image_and_label_batch_cifar10(image, label, batch_size):
  num_preprocess_threads = 1
  images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)
  return images, tf.reshape(label_batch, [batch_size])

def inputs_mineral(batch_size, train=True):
  if train:
    tfrecord_filename = glb('./tfrecords/*train.tfrecord') 
  else:
    tfrecord_filename = glb('./tfrecords/*test.tfrecord') 
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  image, crystal_system, fracture, groups, rock_type = read_data_mineral(filename_queue)

  # data augmentation
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_brightness(image, max_delta=25)
  image = tf.image.random_contrast(image, 0.8, 1.2)
   
  image = tf.random_crop(image, [80, 80, 3])
  image = tf.reshape(image, [1, 80, 80, 3])
  image = tf.image.resize_bicubic(image, [112, 112])
  image = tf.reshape(image, [112, 112, 3])
  image = tf.image.per_image_standardization(image)
    
  # display in tf summary page 
  images, crystal_systems, fractures, groupss, rock_types  = _generate_image_label_batch_mineral(image, crystal_system, fracture, groups, rock_type, batch_size)

  tf.summary.image('mineral image', images)
  return images, crystal_systems, fractures, groupss, rock_types


def inputs_cifar10(batch_size):
  filenames = glb('./cifar10/cifar-10-batches-bin/*.bin')
  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_data_cifar10(filename_queue)
  distorted_image = tf.cast(read_input.uint8image, tf.float32)

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  float_image = tf.image.per_image_standardization(distorted_image)
  #float_image = tf.to_float(distorted_image)

  float_image.set_shape([32,32,3])
  read_input.label.set_shape([1])

  images, labels =  _generate_image_and_label_batch_cifar10(float_image, read_input.label, batch_size)
  images = tf.image.resize_bicubic(images, [112, 112])

  tf.summary.image('cifar image', images)
  return images, labels



