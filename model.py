"""Builds the ring network.
Summary of available functions:
  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import nn 
import inputs
import os
import re
import sys
import tarfile
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                            """ learning rate """)
tf.app.flags.DEFINE_integer('max_steps',  300000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """ training batch size """)

# cifar data url
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def inputs_mineral(batch_size, train=True):
  images, crystal_systems, fractures, groupss, rock_types = inputs.inputs_mineral(batch_size, train)
  return images, crystal_systems, fractures, groupss, rock_types

def inputs_cifar10(batch_size):
  images, labels = inputs.inputs_cifar10(batch_size)
  return images, labels

def inference(images, nr_res=1, nr_downsamples=4, reuse=False, keep_prob=1.0, train_on="mineral"):
  with tf.variable_scope("conv", reuse=reuse):
    x_i = images
    x_i = nn.conv_layer(x_i, 5, 1, 32, "conv_1", nonlinearity=tf.nn.relu)
    x_i = nn.conv_layer(x_i, 3, 1, 32, "conv_2", nonlinearity=tf.nn.relu)
    x_i = nn.max_pool_layer(x_i, 2, 2)
    x_i = nn.conv_layer(x_i, 3, 1, 64, "conv_3", nonlinearity=tf.nn.relu)
    x_i = nn.conv_layer(x_i, 3, 1, 64, "conv_4", nonlinearity=tf.nn.relu)
    x_i = nn.max_pool_layer(x_i, 2, 2)
    x_i = nn.conv_layer(x_i, 3, 1, 128, "conv_5", nonlinearity=tf.nn.relu)
    x_i = nn.conv_layer(x_i, 3, 1, 128, "conv_6", nonlinearity=tf.nn.relu)
    x_i = nn.max_pool_layer(x_i, 2, 2)
    x_i = nn.conv_layer(x_i, 3, 1, 256, "conv_7", nonlinearity=tf.nn.relu)
    x_i = nn.conv_layer(x_i, 3, 1, 256, "conv_8", nonlinearity=tf.nn.relu)
    x_i = nn.max_pool_layer(x_i, 2, 2)
    x_i = nn.fc_layer(x_i, 1024, "fc_0", flat=True, nonlinearity=tf.nn.relu) 
    x_i = tf.nn.dropout(x_i, keep_prob)

  if train_on == "mineral":
    with tf.variable_scope("logits_mineral", reuse=reuse):
      x_i = nn.fc_layer(x_i, 256, "fc_0", nonlinearity=tf.nn.relu) 
      x_i = tf.nn.dropout(x_i, keep_prob)
      crystal_system = nn.fc_layer(x_i, 7, "fc_crystal_system", nonlinearity=None) 
      fracture = nn.fc_layer(x_i, 8*2, "fc_fracture", nonlinearity=None) 
      groups = nn.fc_layer(x_i, 51*2, "fc_groups", nonlinearity=None) 
      rock_type = nn.fc_layer(x_i, 4*2, "fc_rock_type", nonlinearity=None) 
    return crystal_system, fracture, groups, rock_type

  if train_on == "cifar10":
    with tf.variable_scope("logits_cifar"):
      x_i = nn.fc_layer(x_i, 256, "fc_0", nonlinearity=tf.nn.relu) 
      x_i = tf.nn.dropout(x_i, keep_prob)
      label = nn.fc_layer(x_i, 10, "fc_classes", nonlinearity=None) 
    return label

def loss_mineral(crystal_system_t, fracture_t, groups_t, rock_type_t, crystal_system_p, fracture_p, groups_p, rock_type_p, train=True):
  #loss_crystal_system = nn.cross_entropy_binary(crystal_system_t, crystal_system_p) 
  loss_crystal_system = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=crystal_system_t, logits=crystal_system_p))
  tf.summary.scalar('loss crystal system_train_' + str(train), loss_crystal_system)
  loss_fracture = nn.cross_entropy_binary(fracture_t, fracture_p)
  tf.summary.scalar('loss fracture_train_' + str(train), loss_fracture)
  loss_groups = nn.cross_entropy_binary(groups_t, groups_p)
  tf.summary.scalar('loss groups_train_' + str(train), loss_groups)
  loss_rock_type = nn.cross_entropy_binary(rock_type_t, rock_type_p)
  #loss_rock_type = tf.nn.l2_loss(rock_type_t - rock_type_p)
  tf.summary.scalar('loss rock type_train_' + str(train), loss_rock_type)

  #total_loss = loss_rock_type
  #total_loss = loss_crystal_system
  total_loss = loss_crystal_system + loss_fracture + loss_groups + loss_rock_type
  tf.summary.scalar('total loss_train_' + str(train), total_loss)
  return total_loss

def loss_cifar10(logits, labels):
  labels = tf.cast(labels, tf.int64)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  loss = tf.reduce_mean(loss)
  tf.summary.scalar('cifar loss', loss)
  return loss

def train(total_loss, lr, global_step):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step)
   return train_op

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = './cifar10/'
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

