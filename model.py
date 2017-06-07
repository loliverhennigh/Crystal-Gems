"""Builds the ring network.
Summary of available functions:
  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import nn 
import inputs

FLAGS = tf.app.flags.FLAGS

def inputs(batch_size):
  images, crystal_systems, fractures, groupss, rock_types = inputs.flow_inputs(batch_size)
  return images, crystal_systems, fractures, groupss, rock_types

def inference(images, nr_res=2, nr_downsamples=4):

  x_i = images
  for i in xrange(nr_downsamples):
    for j in xrange(nr_res):
      x_i = nn.res_block(x_i, name="block_" + str(i) + "_" + str(j))
    x_i = nn.res_block(x_i, stride=2, name="block_" + str(i) + "_" + str(j))
  x_i = nn.fc_layer(x_i, 1024, "fc_0") 
  x_i = nn.fc_layer(x_i, 51, "fc_0") 
  return x_i

def loss(groupss_p, groupss_t):
  loss = tf.nn.l2_loss(groupss_p - groupss_t)
  tf.summary.scalar('loss', loss)
  return loss

def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
   return train_op

