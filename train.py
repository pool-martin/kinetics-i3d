from __future__ import absolute_import
from __future__ import division
import os

import numpy as np
import tensorflow as tf
from inputs_new import *

import i3d
from config import *

# build the model
def inference(rgb_inputs, flow_inputs):
  with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(
      NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(
      rgb_inputs, is_training=True, dropout_keep_prob=DROPOUT_KEEP_PROB)
  with tf.variable_scope('Flow'):
    flow_model = i3d.InceptionI3d(
        NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    flow_logits, _ = flow_model(
        flow_inputs, is_training=True, dropout_keep_prob=DROPOUT_KEEP_PROB)
  return rgb_logits, flow_logits

# restore the pretrained weights, except for the last layer
def restore():
  # rgb
  rgb_variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
      if 'Logits' in variable.name: # skip the last layer
        continue
      rgb_variable_map[variable.name.replace(':0', '')] = variable
  rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
  # flow
  flow_variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'Flow':
      if 'Logits' in variable.name: # skip the last layer
        continue
      flow_variable_map[variable.name.replace(':0', '')] = variable
  flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
  return rgb_saver, flow_saver


def tower_loss(scope, rgb_inputs, flow_inputs, labels):
  logits = inference(rgb_inputs, flow_inputs)

  return tf.reduce_mean(
             tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    grads_concat = tf.concat(grads, axis=0)
    grads_mean = tf.reduce_mean(grads_concat)

    v = grad_and_vars[0][1]
    average_grads.append((grads_mean, v))
  return average_grads


if __name__ == '__main__':
  pipeline = InputPipeLine(NUM_FRAMES, BATCH_SIZE, FRAME_STRIDE)

  tower_grads = []
  with tf.variable_scope(tf.get_variable_scope()):
    for i in range(NUM_GPUS):
      with tf.device('gpu/:%d' % i):
        with tf.name_scope('tower_%d' % i) as scope:
          rgbs, flows, labels = pipeline.get_batch()
          loss = tower_loss(scope, rgbs, flows, labels)
          tf.get_variable_scope().reuse_variables()
          grads = opt.compute_gradients(loss)
          tower_grads.append(grads)

  grads = average_grads(tower_grads)
  train_op = opt.apply_gradients(grads)


  # saver for fine tuning
  if not os.path.exists(TMPDIR):
    os.mkdir(TMPDIR)
  saver = tf.train.Saver(max_to_keep=SAVER_MAX_TO_KEEP)
  ckpt_path = os.path.join(TMPDIR, 'ckpt')
  if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
      print 'Restoring from:', ckpt.model_checkpoint_path
      saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    else:
      print 'No checkpoint file found, restoring pretrained weights...'
      rgb_saver.restore(sess, CHECKPOINT_PATHS['rgb_imagenet'])
      flow_saver.restore(sess, CHECKPOINT_PATHS['flow_imagenet'])
      print 'Restore Complete.'

    coord, threads = pipeline.start(sess)

    try:
      it = 0
      while it < MAX_ITER and not coord.should_stop():
        if it % DISPLAY_ITER == 0:
          _, loss_val = sess.run([train_op, loss])
          print 'step %d, loss = %.3f' % (it, loss_val)
          # if it > 0:
          # saver.save(sess, ckpt_path + '/model_ckpt', it)
        else:
          sess.run(train_op)
        it += 1
    except KeyboardInterrupt:
      saver.save(sess, os.path.join(ckpt_path, 'model_ckpt'), it)
    coord.request_stop()
    coord.join(threads)
