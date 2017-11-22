import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]

import utils

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1, self.mask1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2, self.mask2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3, self.mask3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4, self.mask4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5, self.mask5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def debuild_full(self):

        deconv_gates = [
            tf.placeholder(dtype=tf.bool, name="{}-deconv-gate".format(layer)) for layer in
            ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
             'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3']]

        outputs = self.op_outputs("pool5")
        activations = self.max_pool_reverse(
            self.fill_filters_with_zeros(outputs, self.max_filter(outputs)), self.mask5)

        outputs = self.op_outputs("conv5_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[16], self.max_filter(outputs)), "conv5_3/filter:0")
        outputs = self.op_outputs("conv5_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[15], self.max_filter(outputs)), "conv5_2/filter:0")
        outputs = self.op_outputs("conv5_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[14], self.max_filter(outputs)), "conv5_1/filter:0")

        outputs = self.op_outputs("pool4")
        activations = self.max_pool_reverse(self.debuild_interlayer(activations, outputs,
                                            deconv_gates[13], self.max_filter(outputs)), self.mask4)
        outputs = self.op_outputs("conv4_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[12], self.max_filter(outputs)), "conv4_3/filter:0")
        outputs = self.op_outputs("conv4_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[11], self.max_filter(outputs)), "conv4_2/filter:0")
        outputs = self.op_outputs("conv4_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[10], self.max_filter(outputs)), "conv4_1/filter:0")

        outputs = self.op_outputs("pool3")
        activations = self.max_pool_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[9], self.max_filter(outputs)), self.mask3)
        outputs = self.op_outputs("conv3_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[8], self.max_filter(outputs)), "conv3_3/filter:0")
        outputs = self.op_outputs("conv3_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[7], self.max_filter(outputs)), "conv3_2/filter:0")
        outputs = self.op_outputs("conv3_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[6], self.max_filter(outputs)), "conv3_1/filter:0")

        outputs = self.op_outputs("pool2")
        activations = self.max_pool_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[5], self.max_filter(outputs)), self.mask2)
        outputs = self.op_outputs("conv2_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[4], self.max_filter(outputs)), "conv2_2/filter:0")
        outputs = self.op_outputs("conv2_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[3], self.max_filter(outputs)), "conv2_1/filter:0")

        outputs = self.op_outputs("pool1")
        activations = self.max_pool_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[2], self.max_filter(outputs)), self.mask1)
        outputs = self.op_outputs("conv1_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[1], self.max_filter(outputs)), "conv1_2/filter:0")
        outputs = self.op_outputs("conv1_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[0], self.max_filter(outputs)), "conv1_1/filter:0")

        return activations, deconv_gates

    def debuild_crop(self):

        deconv_gates = [
            tf.placeholder(dtype=tf.bool, name="{}-deconv-gate".format(layer)) for layer in
            ["conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2", "conv3_1", "conv3_2", "conv3_3", "pool3",
             "conv4_1", "conv4_2", "conv4_3", "pool4", "conv5_1", "conv5_2", "conv5_3"]]

        receptive_field = 212
        outputs = self.op_outputs("pool5")
        activations = self.max_pool_reverse(
            self.mask_max_crop(outputs, 2), self.mask5)

        receptive_field = 196
        outputs = self.op_outputs("conv5_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[16], 3), "conv5_3/filter:0")
        receptive_field = 164
        outputs = self.op_outputs("conv5_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[15], 3), "conv5_2/filter:0")
        receptive_field = 132
        outputs = self.op_outputs("conv5_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[14], 3), "conv5_1/filter:0")

        receptive_field = 100
        outputs = self.op_outputs("pool4")
        activations = self.max_pool_reverse(self.debuild_interlayer_crop(activations, outputs,
                                            deconv_gates[13], 2), self.mask4)
        receptive_field = 92
        outputs = self.op_outputs("conv4_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[12], 3), "conv4_3/filter:0")
        receptive_field = 76
        outputs = self.op_outputs("conv4_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[11], 3), "conv4_2/filter:0")
        receptive_field = 60
        outputs = self.op_outputs("conv4_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[10], 3), "conv4_1/filter:0")

        receptive_field = 44
        outputs = self.op_outputs("pool3")
        activations = self.max_pool_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[9], 2), self.mask3)
        receptive_field = 40
        outputs = self.op_outputs("conv3_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[8], 3), "conv3_3/filter:0")
        receptive_field = 32
        outputs = self.op_outputs("conv3_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[7], 3), "conv3_2/filter:0")
        receptive_field = 24
        outputs = self.op_outputs("conv3_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[6], 3), "conv3_1/filter:0")

        receptive_field = 16
        outputs = self.op_outputs("pool2")
        activations = self.max_pool_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[5], 2), self.mask2)

        receptive_field = 14
        outputs = self.op_outputs("conv2_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[4], 3), "conv2_2/filter:0")

        receptive_field = 10
        outputs = self.op_outputs("conv2_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[3], 3), "conv2_1/filter:0")

        receptive_field = 6
        outputs = self.op_outputs("pool1")
        activations = self.max_pool_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[2], 2), self.mask1)

        receptive_field = 5
        outputs = self.op_outputs("conv1_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[1], 3), "conv1_2/filter:0")

        receptive_field = 3
        outputs = self.op_outputs("conv1_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer_crop(activations, outputs,
                                        deconv_gates[0], 3), "conv1_1/filter:0")

        return activations, deconv_gates

    def debuild_interlayer(self, deconv, activation, deconv_gate, filter_idx):

        return tf.cond(deconv_gate, true_fn=lambda: deconv,
                       false_fn=lambda: self.fill_filters_with_zeros(activation, filter_idx), strict=True)

    def debuild_interlayer_crop(self, deconv, activation, deconv_gate, receptive_field):

        return tf.cond(deconv_gate, true_fn=lambda: deconv,
                       false_fn=lambda: self.mask_max_crop(activation, receptive_field), strict=True)

    def mask_max_crop(self, activations, receptive_field):

        avg_activations = tf.layers.average_pooling2d(activations, [receptive_field, receptive_field], [1, 1],
                                                      padding="VALID")
        spatial_max = tf.reduce_max(avg_activations, axis=[0, 1, 2])

        depth_argmax = tf.argmax(spatial_max, axis=-1)
        spatial_argmax = utils.argmax_2d(avg_activations)

        img_to_crop_location = spatial_argmax[..., depth_argmax]
        print(img_to_crop_location)

        return activations

    def op_outputs(self, name):
        return tf.get_default_graph().get_operation_by_name(name).outputs[0]

    def max_filter(self, values):
        return tf.cast(tf.argmax(tf.reduce_mean(values, axis=(0, 1, 2)), axis=0), tf.int32)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool_with_argmax(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def max_pool_reverse(self, activations, mask, ksize=(1, 2, 2, 1), scope="DeConv2D"):

        with tf.variable_scope(scope):
            input_shape = tf.shape(activations)
            output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            pool_ = tf.reshape(activations, [flat_input_size])
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=mask.dtype),
                                     shape=[input_shape[0], 1, 1, 1])
            b = tf.ones_like(mask) * batch_range
            b1 = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(mask, [flat_input_size, 1])
            ind_ = tf.concat([b1, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, output_shape)

            set_input_shape = activations.get_shape()
            set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2],
                                set_input_shape[3]]
            ret.set_shape(set_output_shape)

            return ret

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding="SAME")

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def conv_reverse(self, activations, conv_weights_name, stride=1):

        input_shape = activations.shape

        conv1_weights = tf.get_default_graph().get_tensor_by_name(conv_weights_name)

        conv1_weights = tf.transpose(conv1_weights, (1, 0, 2, 3))

        output_shape = (input_shape[0].value, input_shape[1].value * stride, input_shape[2].value * stride,
                        conv1_weights.shape[2].value)

        return tf.nn.conv2d_transpose(activations, conv1_weights, output_shape, (1, stride, stride, 1),
                                      padding="SAME", name="DeConv2D")

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the "+" operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def fill_filters_with_zeros(self, tensor, filter_idx):

        zeros = tf.zeros_like(tensor)

        zeros_1 = zeros[..., :filter_idx]
        zeros_2 = zeros[..., filter_idx + 1:]

        zeros = tf.concat([zeros_1, tensor[..., filter_idx:filter_idx + 1], zeros_2], axis=-1)

        return zeros

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
