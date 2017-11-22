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

val_path = "/root/assets/ILSVRC2012_val/images"
annotations_path="resources/ILSVRC12_val_data"

val_imgs = sorted(os.listdir(val_path))
val_imgs = [os.path.join(val_path, path) for path in val_imgs]
val_ann = utils.read_imgnet_labels(annotations_path)

base_dir = "data"
run_dir = new_run_dir(base_dir)

images = tf.placeholder(tf.float32, [1, 224, 224, 3])

vgg = vgg16.Vgg16()
vgg.build(images)

deconv_img, deconv_gates, mask_indexes = vgg.debuild_crop()
