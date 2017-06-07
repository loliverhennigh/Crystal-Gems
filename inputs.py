
import os
import numpy as np
import tensorflow as tf
from glob import glob as glb


FLAGS = tf.app.flags.FLAGS

def read_data(filename_queue):
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
  image = tf.reshape(image, [100, 110, 3])
  crystal_system = tf.to_float(crystal_system)
  fracture = tf.to_float(fracture)
  groups = tf.to_float(groups)
  rock_type = tf.to_float(rock_type)
  image = tf.to_float(image)
  return image, crystal_system, fracture, groups, rock_type

def _generate_image_label_batch(image, crystal_system, fracture, groups, rock_type, batch_size, shuffle=True):
  num_preprocess_threads = 1
  #Create a queue that shuffles the examples, and then
  images, crystal_systems, fractures, groupss, rock_types = tf.train.shuffle_batch(
    [image, crystal_system, fracture, groups, rock_type],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=100 + 3 * batch_size,
    min_after_dequeue=100)
  return images, crystal_systems, fractures, groupss, rock_types

def inputs(batch_size):
  tfrecord_filename = glb('./tfrecords/*.tfrecord') 
  print(tfrecord_filename)
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  image, crystal_system, fracture, groups, rock_type = read_data(filename_queue)

  images, crystal_systems, fractures, groupss, rock_types  = _generate_image_label_batch(image, crystal_system, fracture, groups, rock_type, batch_size)
 
  # display in tf summary page 
  tf.summary.image('mineral image', images)
  return images, crystal_systems, fractures, groupss, rock_types

