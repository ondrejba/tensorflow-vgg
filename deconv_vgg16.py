import cv2, os
import numpy as np
import tensorflow as tf

import vgg16
import utils

def z_norm(img_val):

  return (img_val - np.mean(img_val)) / max(np.std(img_val), 10e-4) * 0.1 + 0.5

def new_run_dir(base):

  idx = 1
  while True:

    path = os.path.join(base, "run{}".format(idx))

    if not os.path.isdir(path):
      os.makedirs(path)
      return path
    else:
      idx += 1

def open_gates_up_to_index(deconv_gates, feed_dict, idx):

  for layer_idx, deconv_gate in enumerate(deconv_gates):

    if layer_idx < idx:
      feed_dict[deconv_gate] = True
    else:
      feed_dict[deconv_gate] = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_dir = "data"
run_dir = new_run_dir(base_dir)

img = utils.load_image("./test_data/tiger.jpeg")

batch = img.reshape((1, 224, 224, 3))

images = tf.placeholder(tf.float32, [1, 224, 224, 3])

vgg = vgg16.Vgg16()
vgg.build(images)

deconv_img, deconv_gates = vgg.debuild()

with tf.Session() as sess:

    feed_dict = {
      images: batch,
      filter_idx: 0
    }

    for layer_idx in range(len(deconv_gates)):

        feed_dict = {
           images: batch,
        }
        open_gates_up_to_index(deconv_gates, feed_dict, layer_idx)

        img_val = sess.run(deconv_img, feed_dict=feed_dict)
        img_val = img_val[0]
        img_val = z_norm(img_val)
        img_val = np.clip(img_val, 0, 1)
        img_val *= 255

        cv2.imwrite(os.path.join(run_dir, "deconv{}.jpg".format(layer_idx)), img_val)
