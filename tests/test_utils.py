import unittest
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

class TestUtils(unittest.TestCase):

  def test_mask_crop(self):

    height = 11
    width = 11
    depth = 3

    image_np = np.ones((height, width, depth))
    image_tf = tf.ones((height, width, depth))

    start_x = 5
    end_x = 8

    start_y = 3
    end_y = 6

    mask_tf = utils.mask_crop(start_x, end_x, start_y, end_y, width, height, depth)
    mask_np = np.zeros_like(image_np)
    mask_np[start_y:end_y, start_x:end_x, :] = 1

    masked_image_tf = image_tf * mask_tf
    masked_image_np = image_np * mask_np

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())

      masked_image_tf_val = sess.run(masked_image_tf)

    np.testing.assert_almost_equal(masked_image_np, masked_image_tf_val)