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

deconv_img, deconv_gates = vgg.debuild()

num_images_per_class = 5
class_counts = {idx: 0 for idx in range(1000)}

with tf.Session() as sess:

    for img_idx, img_path in enumerate(val_imgs):

        cls = val_ann[img_idx]

        if class_counts[cls] < num_images_per_class:

            img = utils.load_image(img_path)
            batch = img.reshape((1, 224, 224, 3))

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

                img_path = os.path.join(run_dir, str(cls), "layer{}.jpg".format(layer_idx))
                cv2.imwrite(img_path, img_val)

            class_counts[cls] += 1
