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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_dir = "data"
run_dir = new_run_dir(base_dir)

img = utils.load_image("./test_data/tiger.jpeg")

batch = img.reshape((1, 224, 224, 3))

images = tf.placeholder(tf.float32, [1, 224, 224, 3])
filter_idx = tf.placeholder(tf.int32)

vgg = vgg16.Vgg16()
vgg.build(images)

activations = tf.get_default_graph().get_operation_by_name("conv3_1/Conv2D").outputs[0]
grad = tf.gradients(activations[..., filter_idx], images)
print(grad)

num_frames = activations.shape[-1].value
deconv_img = vgg.debuild(activations, filter_idx)

with tf.Session() as sess:

    for idx in range(num_frames):
        feed_dict = {
          images: batch,
          filter_idx: idx
        }

        grad_val = sess.run(grad, feed_dict=feed_dict)
        print(grad_val.shape)

        img_val = sess.run(deconv_img, feed_dict=feed_dict)
        img_val = img_val[0]
        img_val = z_norm(img_val)
        img_val = np.clip(img_val, 0, 1)
        img_val *= 255

        cv2.imwrite(os.path.join(run_dir, "deconv{}.jpg".format(idx)), img_val)