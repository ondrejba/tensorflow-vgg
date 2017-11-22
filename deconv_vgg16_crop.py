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


with tf.Session() as sess:

    for img_idx, img_path in enumerate(val_imgs):

        cls = val_ann[img_idx]

        img = utils.load_image(img_path)
        batch = img.reshape((1, 224, 224, 3))

        for layer_idx in range(len(deconv_gates)):

            feed_dict = {
               images: batch,
            }

            open_gates_up_to_index(deconv_gates, feed_dict, layer_idx)

            img_val, mask_indexes_val = sess.run([deconv_img, mask_indexes[layer_idx][1:]], feed_dict=feed_dict)

            receptive_field = mask_indexes[layer_idx][0]
            spatial_idx = mask_indexes_val[0]
            spatial_idx *= receptive_field

            filter_idx = mask_indexes_val[1]

            img_val = img_val[0]
            img_val = img_val[spatial_idx[0] - receptive_field // 2 : spatial_idx[0] + receptive_field // 2,
                              spatial_idx[1] - receptive_field // 2 : spatial_idx[1] + receptive_field // 2]
            img_val = z_norm(img_val)
            img_val = np.clip(img_val, 0, 1)
            img_val *= 255

            save_dir = os.path.join(run_dir, "layer{}".format(layer_idx), "filter{}".format(filter_idx))

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            i = 0
            while True:
                save_path = os.path.join(save_dir, "{}.jpg".format(i))

                if not os.path.isfile(save_path):
                    break
                else:
                    i += 1

            cv2.imwrite(save_path, img_val)
