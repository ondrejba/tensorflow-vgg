"""
source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
"""

import cv2
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import tensorflow as tf

def get_base_image(height=224, width=224, means=None):
  """
  Get base image for filter visualization. Gray image with gaussian noise.
  :param height:        Height of the base image.
  :param width:         Width of the base image.
  :param means:         Means to subtract from the image.
  :return:              Base image as a numpy Tensor.
  """

  background_color = np.float32([200.0, 200.0, 200.0])
  base_image = np.random.normal(background_color, 8, (height, width, 3))

  if means is not None:
    base_image -= means

  return base_image

def create_input_placeholder(means=None):

  input_pl = tf.placeholder(tf.float32, shape=(None, None, 3), name="input")
  input_t = tf.expand_dims(input_pl, axis=0)
  input_t = resize_bilinear(input_t, (224, 224, 3))

  if means is not None:
    input_t = input_t - means

  return input_pl, input_t

def calc_grad_tiled(image, t_grad, session, image_pl, tile_size=512):
  """
  Compute the value of tensor t_grad over the image in a tiled way.
  Random shifts are applied to the image to blur tile boundaries over
  multiple iterations.
  """

  sz = tile_size
  h, w = image.shape[:2]
  sx, sy = np.random.randint(sz, size=2)
  image_shift = np.roll(np.roll(image, sx, 1), sy, 0)
  grad = np.zeros_like(image)

  for y in range(0, max(h - sz // 2, sz), sz):
    for x in range(0, max(w - sz // 2, sz), sz):
      sub = image_shift[y:y + sz, x:x + sz]
      g = session.run(t_grad, {image_pl: sub})
      grad[y:y + sz, x:x + sz] = g

  return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_multiscale(objective, image_pl, session, resize_op, resize_image_pl, resize_shape_pl,iter_n=10, step=1.0,
                      octave_n=3, octave_scale=1.4, means=None):

  # compute a scalar value to optimize and derive its gradient
  score = tf.reduce_mean(objective)
  gradient = tf.gradients(score, image_pl)[0]

  image = get_base_image(224, 224, means=means)

  for octave in range(octave_n):

    if octave > 0:
      hw = np.int32(np.float32(image.shape[:2]) * octave_scale)
      image = session.run(resize_op, feed_dict={
        resize_image_pl: image,
        resize_shape_pl: hw
      })

    for i in range(iter_n):

      g = calc_grad_tiled(image, gradient, session, image_pl)
      # normalizing the gradient, so the same step size should work
      g /= g.std() + 1e-8
      image += g * step

  return image

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    """ Split the image into lo and hi frequency components. """
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    """ Build Laplacian pyramid with n splits. """
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    """ Merge Laplacian pyramid. """
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    """ Normalize image by making its standard deviation = 1.0. """
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    """ Perform the Laplacian pyramid normalization. """
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

def setup_resize():

  resize_image_pl = tf.placeholder(tf.float32, shape=(None, None, 3), name="resize_image_pl")
  resize_shape_pl = tf.placeholder(tf.int32, shape=(2,), name="resize_shape_pl")
  resize_op = tf.image.resize_bilinear(tf.expand_dims(resize_image_pl, 0), resize_shape_pl)[0, ...]

  return resize_op, resize_image_pl, resize_shape_pl

def setup_lapnorm(scale_n=4):

  lapnorm_pl = tf.placeholder(tf.float32, shape=(None, None, 3), name="lapnorm_pl")
  lapnorm = lap_normalize(lapnorm_pl, scale_n=scale_n)

  return lapnorm, lapnorm_pl

def render_lapnorm(objective, session, image_pl, lap_norm, lap_norm_pl, resize_op, resize_image_pl,
                   resize_shape_pl, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, means=None):

  score = tf.reduce_mean(objective)
  gradient = tf.gradients(score, image_pl)[0]
  image = get_base_image(224, 224, means=means)

  for octave in range(octave_n):

    if octave > 0:
      hw = np.int32(np.float32(image.shape[:2]) * octave_scale)
      image = session.run(resize_op, feed_dict={
        resize_image_pl: image,
        resize_shape_pl: hw
      })

    for i in range(iter_n):

      g = calc_grad_tiled(image, gradient, session, image_pl)
      g = session.run(lap_norm, feed_dict={
        lap_norm_pl: g
      })
      image += g * step

  return image

def normalize_image(image, s=0.1):
  """ Normalize the image range for visualization. """
  new_image = image / 255
  return (new_image - new_image.mean()) / max(new_image.std(), 1e-4) * s + 0.5

def save_image(filename, image):

  image = np.clip(image, 0, 1)
  image *= 255

  cv2.imwrite(filename, image)

def show_image(image, verbose=False):

  if verbose:
    print("output statistics:")
    print("mean:", np.mean(image))
    print("max: ", np.max(image))
    print("min: ", np.min(image))

  image = np.clip(image, 0, 1)

  plt.imshow(image)
  plt.show()