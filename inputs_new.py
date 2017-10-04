import tensorflow as tf
import numpy as np
import os
import threading
# import train code here

FRAME_DATA_PATH = '/media/6TB/Videos/UCF-101-frames'
CROP_SIZE = 224

def enqueue(sess, enqueue_op, num_frames, cls_dict):
  folders = [folder for folder in os.listdir(FRAME_DATA_PATH) if folder.startswith('v')]
  while True:
    index = np.random.randint(0, len(folders))
    prefix = os.path.join(FRAME_DATA_PATH, folders[index])
    cls_name = folders[index].split('_')[1]
    sorted_list = np.sort(os.listdir(prefix))
    imgs = [os.path.join(prefix, img) for img in sorted_list if img.startswith('img')]
    flow_xs = [os.path.join(prefix, flow) for flow in sorted_list if flow.startswith('flow_x')]
    flow_ys = [os.path.join(prefix, flow) for flow in sorted_list if flow.startswith('flow_y')]
    assert len(imgs) == len(flow_xs)
    assert len(imgs) == len(flow_ys)
    if num_frames <= len(imgs):
      begin = np.random.randint(0, len(imgs) - num_frames + 1)
    else:
      begin = 0
      ori_len = len(imgs)
      while num_frames > len(imgs):
        for i in range(ori_len):
          imgs.append(imgs[i])
          flow_xs.append(flow_xs[i])
          flow_ys.append(flow_ys[i])
          if len(imgs) == num_frames:
            break

    imgs_out = imgs[begin:begin + num_frames]
    flow_xs_out = flow_xs[begin:begin + num_frames]
    flow_ys_out = flow_ys[begin:begin + num_frames]
    # print imgs_out, flow_xs_out, flow_ys_out, cls_dict[cls_name]
    sess.run(enqueue_op, {rgb: imgs_out, flow_x: flow_xs_out, flow_y: flow_ys_out, label: cls_dict[cls_name]})


def inputs(rgb, flow_x, flow_y, label, batch_size):
  num_frames = rgb.get_shape()[0]
  queue = tf.FIFOQueue(capacity=32, dtypes=[tf.string, tf.string, tf.string, tf.int32], shapes=[[num_frames],[num_frames],[num_frames],[]])
  enqueue_op = queue.enqueue([rgb, flow_x, flow_y, label])
  item = queue.dequeue()
  rgb_frames = []
  flow_x_frames = []
  flow_y_frames = []
  for i in range(num_frames):
    rgb_frames.append(tf.image.decode_jpeg(tf.read_file(item[0][i]), channels=3))
    flow_x_frames.append(tf.image.decode_jpeg(tf.read_file(item[1][i]), channels=1))
    flow_y_frames.append(tf.image.decode_jpeg(tf.read_file(item[2][i]), channels=1))
  
  output_rgb = tf.stack(rgb_frames, axis=0)
  tmp_flow_x = tf.stack(flow_x_frames, axis=0)
  tmp_flow_y = tf.stack(flow_y_frames, axis=0)
  output_flow = tf.concat([tmp_flow_x, tmp_flow_y], axis=3)

  # random crop
  rgb_flow_concat = tf.concat([output_rgb, output_flow], axis=3)
  crop_concat = tf.random_crop(rgb_flow_concat, [int(num_frames), CROP_SIZE, CROP_SIZE, 5])
  output_rgb = crop_concat[:,:,:,:3]
  output_flow = crop_concat[:,:,:,3:]
 
  # rescale
  output_rgb = tf.cast(output_rgb, tf.float32)
  output_flow = tf.cast(output_flow, tf.float32)
  output_rgb = output_rgb * 2 / 255.0 - 1
  output_flow = output_flow * 2 / 255.0 - 1

  label = tf.cast(item[3], tf.int32)
  rgbs, flows, labels = tf.train.batch([output_rgb, output_flow, label], batch_size=batch_size)
  return enqueue_op, rgbs, flows, labels


def build_cls_dict():
  cls_dict = {}
  folders = np.sort([f for f in os.listdir(FRAME_DATA_PATH) if f.startswith('v')])
  l = 0
  for folder in folders:
    cls_name = folder.split('_')[1]
    if not cls_name in cls_dict:
      cls_dict[cls_name] = l
      l += 1
  return cls_dict


# if __name__ == '__main__':
#   with tf.Graph().as_default() as g:
#     # placeholders for input queue
#     rgb = tf.placeholder(tf.string, shape=[NUM_FRAMES])
#     flow_x = tf.placeholder(tf.string, shape=[NUM_FRAMES])
#     flow_y = tf.placeholder(tf.string, shape=[NUM_FRAMES])
#     label = tf.placeholder(tf.int32)
#     # cls_dict maps class names to integer labels
#     cls_dict = build_cls_dict()

#     enqueue_op, batch = inputs(rgb, flow_x, flow_y, label, 10)
#     with tf.Session() as sess:
#       enqueue_thread = threading.Thread(target=enqueue, args=[sess, enqueue_op, NUM_FRAMES, cls_dict])
#       enqueue_thread.isDaemon()
#       enqueue_thread.start()
      
#       coord = tf.train.Coordinator()
#       threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#       print batch 
#       res = sess.run(batch[2])
#       print res
 
#       coord.request_stop()
#       coord.join(threads)
