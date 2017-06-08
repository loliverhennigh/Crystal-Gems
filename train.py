import numpy as np
import tensorflow as tf
import os

import model
import inputs 
import time

FLAGS = tf.app.flags.FLAGS

TRAIN_DIR = "./checkpoints/run_0"

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # global step counter
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # make inputs mineral
    images_train, crystal_systems_train_t, fractures_train_t, groupss_train_t, rock_types_train_t = inputs.inputs_mineral(FLAGS.batch_size, train=True) 
    images_test, crystal_systems_test_t, fractures_test_t, groupss_test_t, rock_types_test_t = inputs.inputs_mineral(FLAGS.batch_size, train=False) 

    # make inputs cifar
    images_cifar, labels_cifar = inputs.inputs_cifar10(FLAGS.batch_size) 

    # create network train
    crystal_systems_train_p, fractures_train_p, groupss_train_p, rock_types_train_p = model.inference(images_train, keep_prob=0.5) 
    crystal_systems_test_p, fractures_test_p, groupss_test_p, rock_types_test_p = model.inference(images_test, reuse=True) 

    # create network cifar
    logits_cifar = model.inference(images_cifar, train_on="cifar10", reuse=True) 

    # calc error mineral
    error_train = model.loss_mineral(crystal_systems_train_t, fractures_train_t, groupss_train_t, rock_types_train_t, crystal_systems_train_p, fractures_train_p, groupss_train_p, rock_types_train_p, train=True)
    error_test = model.loss_mineral(crystal_systems_test_t, fractures_test_t, groupss_test_t, rock_types_test_t, crystal_systems_test_p, fractures_test_p, groupss_test_p, rock_types_test_p, train=False)

    # calc error cifar
    error_cifar = model.loss_cifar10(logits_cifar, labels_cifar)

    # train hopefuly 
    train_op_mineral = model.train(error_train, FLAGS.learning_rate, global_step)
    train_op_cifar = model.train(error_cifar, FLAGS.learning_rate, global_step)

    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   
    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    sess.run(init)
 
    # init from checkpoint
    saver_restore = tf.train.Saver(variables)
    ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if ckpt is not None:
      print("init from " + TRAIN_DIR)
      try:
         saver_restore.restore(sess, ckpt.model_checkpoint_path)
      except:
         tf.gfile.DeleteRecursively(TRAIN_DIR)
         tf.gfile.MakeDirs(TRAIN_DIR)
         print("there was a problem using variables in checkpoint, random init will be used instead")

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)

    # calc number of steps left to run
    run_steps = FLAGS.max_steps - int(sess.run(global_step))
    _ , loss_mineral = sess.run([train_op_mineral, error_train])
    for step in xrange(run_steps):
      current_step = sess.run(global_step)
      t = time.time()
      #print(sess.run(images_train))
      if current_step > 40000:
        _, loss_mineral = sess.run([train_op_mineral, error_train])
        #loss_cifar = sess.run([error_cifar])
      else:
        loss_mineral = sess.run(error_train)
      _ , loss_cifar = sess.run([train_op_cifar, error_cifar])
      #print(sess.run(logits_cifar))
      #print(sess.run(labels_cifar))
      elapsed = time.time() - t

      assert not np.isnan(loss_mineral), 'Model diverged with loss = NaN'

      if current_step % 100 == 1:
        #print("groupss_p_out " + str(groupss_p_out))
        #print("groupss_t_out " + str(groupss_t_out))
        print("loss mineral value at " + str(loss_mineral))
        print("loss cifar value at " + str(loss_cifar))
        print("time per batch is " + str(elapsed))

      if current_step % 1000 == 1:
        loss_test_value = sess.run(error_test)
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, current_step) 
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  model.maybe_download_and_extract()
  train()

if __name__ == '__main__':
  tf.app.run()
