import argparse, os
import tensorflow as tf
import numpy as np

import vgg16
import deep_dream

def main(args):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  input_pl, input_t = deep_dream.create_input_placeholder()
  lapnorm, lapnorm_pl = deep_dream.setup_lapnorm(scale_n=args.num_scales)
  resize_op, resize_image_pl, resize_shape_pl = deep_dream.setup_resize()

  model = vgg16.Vgg16()

  model.build(input_t / 255)

  objective_names = ['conv1_1/Conv2D', 'conv1_2/Conv2D', 'pool1', 'conv2_1/Conv2D', 'conv2_2/Conv2D', 'pool2', 'conv3_1/Conv2D', 'conv3_2/Conv2D', 'conv3_3/Conv2D', 'pool3',
                'conv4_1/Conv2D', 'conv4_2/Conv2D', 'conv4_3/Conv2D', 'pool4', 'conv5_1/Conv2D', 'conv5_2/Conv2D', 'conv5_3/Conv2D']
  objectives = [tf.get_default_graph().get_operation_by_name(name).outputs[0] for name in objective_names]

  with tf.Session() as sess:

    for objective, name in zip(objectives, objective_names):

      dir_path = os.path.join(args.save_dir, name)

      if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

      for filter_idx in range(objective.shape[-1]):

        save_path = os.path.join(dir_path, "{:d}.jpg".format(filter_idx))

        if not os.path.isfile(save_path) or args.override:
          if not args.disable_lapnorm:
            image = deep_dream.render_lapnorm(objective[..., filter_idx], sess, input_pl, lapnorm, lapnorm_pl,
                                              resize_op, resize_image_pl, resize_shape_pl)
          else:
            image = deep_dream.render_multiscale(objective[..., filter_idx], input_pl, sess, resize_op, resize_image_pl,
                                               resize_shape_pl)
          deep_dream.save_image(save_path, deep_dream.normalize_image(image))

parser = argparse.ArgumentParser("Create deep dream visualization for a given objective.")

parser.add_argument("-s", "--num-scales", default=4, type=int)
parser.add_argument("--disable-lapnorm", default=False, action="store_true")
parser.add_argument("--save-dir", default="data/deep_dream/")
parser.add_argument("--override", default=False, action="store_true")

parsed = parser.parse_args()
main(parsed)