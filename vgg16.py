import os, inspect, math

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

    def debuild_full(self, use_biases=True):

        deconv_gates = [
            tf.placeholder(dtype=tf.bool, name="{}-deconv-gate".format(layer)) for layer in
            ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
             'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3']]

        outputs = self.op_outputs("pool5")
        activations = self.max_pool_reverse(
            self.fill_filters_with_zeros(outputs, self.max_filter(outputs)), self.mask5)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv5_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[16], self.max_filter(outputs)), "conv5_3/filter:0",
                                        biases_name="conv5_3/biases:0", use_biases=use_biases)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv5_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[15], self.max_filter(outputs)), "conv5_2/filter:0",
                                        biases_name="conv5_2/biases:0", use_biases=use_biases)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv5_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[14], self.max_filter(outputs)), "conv5_1/filter:0",
                                        biases_name="conv5_1/biases:0", use_biases=use_biases)

        outputs = self.op_outputs("pool4")
        activations = self.max_pool_reverse(self.debuild_interlayer(activations, outputs,
                                            deconv_gates[13], self.max_filter(outputs)), self.mask4)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv4_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[12], self.max_filter(outputs)), "conv4_3/filter:0",
                                        biases_name="conv4_3/biases:0", use_biases=use_biases)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv4_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[11], self.max_filter(outputs)), "conv4_2/filter:0",
                                        biases_name="conv4_2/biases:0", use_biases=use_biases)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv4_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[10], self.max_filter(outputs)), "conv4_1/filter:0",
                                        biases_name="conv4_1/biases:0", use_biases=use_biases)

        outputs = self.op_outputs("pool3")
        activations = self.max_pool_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[9], self.max_filter(outputs)), self.mask3)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv3_3/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[8], self.max_filter(outputs)), "conv3_3/filter:0",
                                        biases_name="conv3_3/biases:0", use_biases=use_biases)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv3_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[7], self.max_filter(outputs)), "conv3_2/filter:0",
                                        biases_name="conv3_2/biases:0", use_biases=use_biases)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv3_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[6], self.max_filter(outputs)), "conv3_1/filter:0",
                                        biases_name="conv3_1/biases:0", use_biases=use_biases)

        outputs = self.op_outputs("pool2")
        activations = self.max_pool_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[5], self.max_filter(outputs)), self.mask2)
        
        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv2_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[4], self.max_filter(outputs)), "conv2_2/filter:0",
                                        biases_name="conv2_2/biases:0", use_biases=use_biases)

        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv2_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[3], self.max_filter(outputs)), "conv2_1/filter:0",
                                        biases_name="conv2_1/biases:0", use_biases=use_biases)

        outputs = self.op_outputs("pool1")
        activations = self.max_pool_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[2], self.max_filter(outputs)), self.mask1)

        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv1_2/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[1], self.max_filter(outputs)), "conv1_2/filter:0",
                                        biases_name="conv1_2/biases:0", use_biases=use_biases)

        activations = tf.nn.relu(activations)

        outputs = self.op_outputs("conv1_1/Conv2D")
        activations = self.conv_reverse(self.debuild_interlayer(activations, outputs,
                                        deconv_gates[0], self.max_filter(outputs)), "conv1_1/filter:0",
                                        biases_name="conv1_1/biases:0", use_biases=use_biases)

        return activations, deconv_gates

    def debuild_crop(self, use_biases=True, mask=True):

        deconv_gates = [
            tf.placeholder(dtype=tf.bool, name="{}-deconv-gate".format(layer)) for layer in
            ["conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2", "conv3_1", "conv3_2", "conv3_3", "pool3",
             "conv4_1", "conv4_2", "conv4_3", "pool4", "conv5_1", "conv5_2", "conv5_3"]]
        mask_indexes = []

        # BLOCK 5
        receptive_field = 212
        outputs = self.op_outputs("pool5")
        masked, idxs, filter_idx = self.mask_max_crop(outputs, mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.max_pool_reverse(masked, self.mask5)

        activations = tf.nn.relu(activations)

        receptive_field = 196

        outputs = self.op_outputs("conv5_3/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[16], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv5_3/filter:0", biases_name="conv5_3/biases:0",
                                        use_biases=use_biases)

        activations = tf.nn.relu(activations)

        receptive_field = 164
        outputs = self.op_outputs("conv5_2/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[15], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv5_2/filter:0", biases_name="conv5_2/biases:0",
                                        use_biases=use_biases)

        activations = tf.nn.relu(activations)

        receptive_field = 132
        outputs = self.op_outputs("conv5_1/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[14], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv5_1/filter:0", biases_name="conv5_1/biases:0",
                                        use_biases=use_biases)

        # BLOCK 4
        receptive_field = 100
        outputs = self.op_outputs("pool4")
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[13], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.max_pool_reverse(masked, self.mask4)

        activations = tf.nn.relu(activations)

        receptive_field = 92
        outputs = self.op_outputs("conv4_3/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[12], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv4_3/filter:0", biases_name="conv4_3/biases:0",
                                        use_biases=use_biases)

        activations = tf.nn.relu(activations)

        receptive_field = 76
        outputs = self.op_outputs("conv4_2/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[11], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv4_2/filter:0", biases_name="conv4_2/biases:0",
                                        use_biases=use_biases)

        activations = tf.nn.relu(activations)

        receptive_field = 60
        outputs = self.op_outputs("conv4_1/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[10], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv4_1/filter:0", biases_name="conv4_1/biases:0",
                                        use_biases=use_biases)

        # BLOCK 3
        receptive_field = 44
        outputs = self.op_outputs("pool3")
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[9], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.max_pool_reverse(masked, self.mask3)

        activations = tf.nn.relu(activations)

        receptive_field = 40
        outputs = self.op_outputs("conv3_3/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[8], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv3_3/filter:0", biases_name="conv3_3/biases:0",
                                        use_biases=use_biases)

        activations = tf.nn.relu(activations)

        receptive_field = 32
        outputs = self.op_outputs("conv3_2/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[7], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv3_2/filter:0", biases_name="conv3_2/biases:0",
                                        use_biases=use_biases)

        activations = tf.nn.relu(activations)

        receptive_field = 24
        outputs = self.op_outputs("conv3_1/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[6], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv3_1/filter:0", biases_name="conv3_1/biases:0",
                                        use_biases=use_biases)

        # BLOCK 2
        receptive_field = 16
        outputs = self.op_outputs("pool2")
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[5], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.max_pool_reverse(masked, self.mask2)

        activations = tf.nn.relu(activations)

        receptive_field = 14
        outputs = self.op_outputs("conv2_2/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[4], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv2_2/filter:0", biases_name="conv2_2/biases:0",
                                        use_biases=use_biases)

        activations = tf.nn.relu(activations)

        receptive_field = 10
        outputs = self.op_outputs("conv2_1/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[3], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv2_1/filter:0", biases_name="conv2_1/biases:0",
                                        use_biases=use_biases)

        # BLOCK 1
        receptive_field = 6
        outputs = self.op_outputs("pool1")
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[2], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.max_pool_reverse(masked, self.mask1)

        activations = tf.nn.relu(activations)

        receptive_field = 5
        outputs = self.op_outputs("conv1_2/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[1], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv1_2/filter:0", biases_name="conv1_2/biases:0",
                                        use_biases=use_biases)

        activations = tf.nn.relu(activations)

        receptive_field = 3
        outputs = self.op_outputs("conv1_1/Conv2D", use_biases=use_biases)
        masked, idxs, filter_idx = self.debuild_interlayer_crop(activations, outputs, deconv_gates[0], mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        activations = self.conv_reverse(masked, "conv1_1/filter:0", biases_name="conv1_1/biases:0",
                                        use_biases=use_biases)

        mask_indexes = list(reversed(mask_indexes))

        return activations, deconv_gates, mask_indexes

    def degrad_crop(self, input_tensor, mask=False):

        mask_indexes = []
        outputs_list = []

        # BLOCK 5
        receptive_field = 212
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "pool5", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 196
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv5_3/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 164
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv5_2/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 132
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv5_1/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        # BLOCK 4
        receptive_field = 100
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "pool4", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 92
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv4_3/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 76
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv4_2/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 60
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv4_1/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        # BLOCK 3
        receptive_field = 44
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "pool3", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 40
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv3_3/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 32
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv3_2/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 24
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv3_1/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        # BLOCK 2
        receptive_field = 16
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "pool2", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 14
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv2_2/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 10
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv2_1/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        # BLOCK 1
        receptive_field = 6
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "pool1", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 5
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv1_2/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        receptive_field = 3
        outputs, idxs, filter_idx = self.degrad_layer_crop(input_tensor, "conv1_1/Conv2D", mask=mask)
        mask_indexes.append((receptive_field, idxs, filter_idx))
        outputs_list.append(outputs)

        mask_indexes = list(reversed(mask_indexes))
        outputs_list = list(reversed(outputs_list))

        return outputs_list, mask_indexes

    def debuild_interlayer(self, deconv, activation, deconv_gate, filter_idx):

        return tf.cond(deconv_gate, true_fn=lambda: deconv,
                       false_fn=lambda: self.fill_filters_with_zeros(activation, filter_idx), strict=True)

    def debuild_interlayer_crop(self, deconv, activation, deconv_gate, mask=True):

        masked, idxs, filter_idx = self.mask_max_crop(activation, mask=mask)

        return tf.cond(deconv_gate, true_fn=lambda: deconv,
                       false_fn=lambda: masked, strict=True), idxs, filter_idx

    def degrad_layer_crop(self, input_tensor, op_name, mean_filter_activation=True, original_size=224, mask=False):

        outputs = self.op_outputs(op_name)

        if mean_filter_activation:
            spatial_reduce = tf.reduce_mean(outputs, axis=[0, 1, 2])
        else:
            spatial_reduce = tf.reduce_max(outputs, axis=[0, 1, 2])

        depth_argmax = tf.cast(tf.argmax(spatial_reduce, axis=-1), tf.int32)


        ratio = int(original_size / outputs.shape[1].value)
        spatial_argmax = utils.argmax_2d(outputs)[0, :, depth_argmax]

        if mask:
            mask = tf.zeros((outputs.shape[1].value, outputs.shape[2].value))
            delta = tf.SparseTensor([tf.cast(spatial_argmax, tf.int64)], [1.0], [outputs.shape[1].value, outputs.shape[2].value])
            mask += tf.sparse_tensor_to_dense(delta)

            outputs = tf.multiply(outputs, tf.cast(tf.stack([tf.stack([mask for _ in range(outputs.shape[3].value)], axis=-1)], axis=0), tf.float32))

        grads = tf.gradients(outputs[..., depth_argmax], input_tensor, outputs[..., depth_argmax])[0]
        spatial_argmax = tf.multiply(spatial_argmax, ratio)

        return grads, spatial_argmax, depth_argmax

    def mask_max_crop(self, activations, original_size=224, mean_filter_activation=True, mask=True):

        ratio = int(original_size / activations.shape[1].value)

        if mean_filter_activation:
            spatial_reduce = tf.reduce_mean(activations, axis=[0, 1, 2])
        else:
            spatial_reduce = tf.reduce_max(activations, axis=[0, 1, 2])

        depth_argmax = tf.cast(tf.argmax(spatial_reduce, axis=-1), tf.int32)
        spatial_argmax = utils.argmax_2d(activations)[0, :, depth_argmax]

        if mask:
            mask = tf.zeros((activations.shape[1].value, activations.shape[2].value), dtype=tf.float32)
            delta = tf.SparseTensor([tf.cast(spatial_argmax, tf.int64)], [1.0], [activations.shape[1].value, activations.shape[2].value])
            mask += tf.sparse_tensor_to_dense(delta)

            activations = tf.multiply(activations, tf.cast(tf.stack([tf.stack([mask] * activations.shape[3].value, axis=-1)], axis=0), tf.float32))

        activations = self.fill_filters_with_zeros(activations, depth_argmax)

        spatial_argmax = tf.multiply(spatial_argmax, ratio)

        return activations, spatial_argmax, depth_argmax

    def get_all_outputs(self):

        outputs_list = []

        outputs_list.append(self.op_outputs("pool5"))
        outputs_list.append(self.op_outputs("conv5_3/Conv2D"))
        outputs_list.append(self.op_outputs("conv5_2/Conv2D"))
        outputs_list.append(self.op_outputs("conv5_1/Conv2D"))

        outputs_list.append(self.op_outputs("pool4"))
        outputs_list.append(self.op_outputs("conv4_3/Conv2D"))
        outputs_list.append(self.op_outputs("conv4_2/Conv2D"))
        outputs_list.append(self.op_outputs("conv4_1/Conv2D"))

        outputs_list.append(self.op_outputs("pool3"))
        outputs_list.append(self.op_outputs("conv3_3/Conv2D"))
        outputs_list.append(self.op_outputs("conv3_2/Conv2D"))
        outputs_list.append(self.op_outputs("conv3_1/Conv2D"))

        outputs_list.append(self.op_outputs("pool2"))
        outputs_list.append(self.op_outputs("conv2_2/Conv2D"))
        outputs_list.append(self.op_outputs("conv2_1/Conv2D"))

        outputs_list.append(self.op_outputs("pool1"))
        outputs_list.append(self.op_outputs("conv1_2/Conv2D"))
        outputs_list.append(self.op_outputs("conv1_1/Conv2D"))

        outputs_list = list(reversed(outputs_list))

        return outputs_list

    def get_filter_reduces(self, filters, outputs_list, reduce_max=False):

        filter_reduces = {}

        for layer_idx, filters_list in filters.items():

            filter_reduces[layer_idx] = []

            for filter_idx, in filters_list:

                if reduce_max:
                    filter_reduce = tf.reduce_max(outputs_list[layer_idx][..., filter_idx], axis=(0, 1, 2))
                else:
                    filter_reduce = tf.reduce_mean(outputs_list[layer_idx][..., filter_idx], axis=(0, 1, 2))

                filter_reduces.append(filter_reduce)

        return filter_reduces

    def op_outputs(self, name, use_biases=False):
        if use_biases:
            return tf.get_default_graph().get_operation_by_name(name + "_biases").outputs[0]
        else:
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
            bias = tf.nn.bias_add(conv, conv_biases, name="Conv2D_biases")

            relu = tf.nn.relu(bias)
            return relu

    def conv_reverse(self, activations, conv_weights_name, stride=1, biases_name=None, use_biases=False):

        input_shape = activations.shape

        conv1_weights = tf.get_default_graph().get_tensor_by_name(conv_weights_name)

        conv1_weights = tf.transpose(conv1_weights, (1, 0, 3, 2))
        print(conv1_weights)
        print(activations)

        output_shape = (input_shape[0].value, input_shape[1].value * stride, input_shape[2].value * stride,
                        conv1_weights.shape[2].value)

        if use_biases:
            biases = tf.get_default_graph().get_tensor_by_name(biases_name)
            activations -= biases

        output = tf.nn.conv2d(activations, conv1_weights, [1, 1, 1, 1], padding="SAME", name="DeConv2D")

        #output = tf.nn.conv2d_transpose(activations, conv1_weights, output_shape, (1, stride, stride, 1),
        #                              padding="SAME", name="DeConv2D")

        return output

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
