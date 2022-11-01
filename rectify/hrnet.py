import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras


class BottleNeckBlock(keras.layers.Layer):
    def __init__(self, in_channel=64, out_channel=256, trans=False):
        super(BottleNeckBlock, self).__init__()
        self.relu = keras.layers.ReLU()
        self.norm1 = keras.layers.BatchNormalization()
        self.norm2 = keras.layers.BatchNormalization()
        self.norm3 = keras.layers.BatchNormalization()
        self.conv1 = keras.layers.Conv2D(filters=in_channel, kernel_size=1, strides=1, padding='same')
        self.conv2 = keras.layers.Conv2D(filters=in_channel, kernel_size=3, strides=1, padding='same')
        self.conv3 = keras.layers.Conv2D(filters=out_channel, kernel_size=1, strides=1, padding='same')

        self.block = keras.Sequential([self.conv1, self.norm1, self.relu,
                                       self.conv2, self.norm2, self.relu,
                                       self.conv3, self.norm3])

        self.trans = trans
        if self.trans:
            self.trans_conv = keras.layers.Conv2D(filters=out_channel, kernel_size=1, strides=1, padding='same')
            self.trans_norm = keras.layers.BatchNormalization()
            self.trans_block = keras.Sequential([self.trans_conv, self.trans_norm])

    def call(self, inputs):
        residual = inputs

        outputs = self.block(inputs)
        if self.trans:
            residual = self.trans_block(residual)

        _outputs = tf.add(residual, outputs)
        outputs = self.relu(_outputs)

        return outputs


class ExchangeBlock(keras.Model):
    def __init__(self, layers, within=False, filter_size=18):      # [32,64,128,256] OR [128, 256, 512, 1024]
        super(ExchangeBlock, self).__init__()
        filter_size = [filter_size*(2**i) for i in range(4)]
        self.num_layers = layers
        self.within = within

        self.conv_layers = []
        for i in range(layers):
            one_layer = []
            for j in range(layers):
                if i == j:
                    one_layer.append(None)
                elif i < j:
                    block = keras.Sequential([keras.layers.Conv2D(filters=filter_size[i], kernel_size=1, strides=1, padding='valid'),
                                              keras.layers.BatchNormalization()])
                    one_layer.append(block)
                else:
                    mul_layers = []
                    for k in range(i-j):
                        conv = keras.layers.Conv2D(filters=filter_size[i], kernel_size=3, strides=2, padding='same')
                        norm = keras.layers.BatchNormalization()
                        mul_layers.append(conv)
                        mul_layers.append(norm)
                        if k != i-j-1:
                            mul_layers.append(keras.layers.ReLU())
                    block = keras.Sequential(mul_layers)
                    one_layer.append(block)
            self.conv_layers.append(one_layer)

        if not self.within:
            self.last_conv = keras.layers.Conv2D(filters=filter_size[layers], kernel_size=3, strides=2, padding='same')
            self.last_norm = keras.layers.BatchNormalization()
            self.subnet_block = keras.Sequential([self.last_conv, self.last_norm, keras.layers.ReLU()])

    def call(self, inputs):
        outputs = []
        for i in range(self.num_layers):
            curr_net = 0
            for j in range(self.num_layers):
                if i == j:  # current resolution
                    temp_net = inputs[i]
                elif i < j:  # i is higher resolution
                    _, x, y, _ = inputs[i].get_shape()
                    temp_net = self.conv_layers[i][j](inputs[j])
                    temp_net = tf.image.resize(temp_net, [x, y])
                else:  # i is lower resolution
                    _, x, y, _ = inputs[i].get_shape()
                    temp_net = self.conv_layers[i][j](inputs[j])
                curr_net = tf.add(temp_net, curr_net)
            curr_net = tf.nn.relu(curr_net)
            outputs.append(curr_net)

        if not self.within:
            subnetwork = self.subnet_block(outputs[-1])
            outputs.append(subnetwork)

        return outputs


class ParallelBlock(keras.Model):
    def __init__(self, filter_size, layers=4):
        super(ParallelBlock, self).__init__()
        self.num_layers = layers
        self.conv_layers = []
        for i in range(layers):
            conv1 = keras.layers.Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same')
            conv2 = keras.layers.Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same')
            block = keras.Sequential([conv1, keras.layers.BatchNormalization(), keras.layers.ReLU(),
                                      conv2, keras.layers.BatchNormalization()])
            self.conv_layers.append(block)

    def call(self, input):
        outputs = input
        for i in range(self.num_layers):
            residual = outputs
            output = self.conv_layers[i](outputs)
            output = tf.add(residual, output)
            outputs = tf.nn.relu(output)
        return outputs


class Stage1(keras.layers.Layer):
    def __init__(self, filter_size=18):
        super(Stage1, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.batch1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.batch2 = keras.layers.BatchNormalization()
        self.init_block = keras.Sequential([self.conv1, self.batch1, keras.layers.ReLU(),
                                            self.conv2, self.batch2, keras.layers.ReLU()])

        self.bottleneck_block = keras.Sequential([BottleNeckBlock(trans=True),
                                                  BottleNeckBlock(),
                                                  BottleNeckBlock(),
                                                  BottleNeckBlock()])

        self.conv3 = keras.layers.Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same')
        self.batch3 = keras.layers.BatchNormalization()
        self.out_block = keras.Sequential([self.conv3, self.batch3, keras.layers.ReLU()])

        self.full_block = keras.Sequential([self.init_block, self.bottleneck_block, self.out_block])

        self.exchange = ExchangeBlock(layers=1)

    def call(self, inputs):
        outputs = self.full_block(inputs)
        outputs = [outputs]
        outputs = self.exchange(outputs)
        return outputs


class Stage2(keras.layers.Layer):
    def __init__(self, width=18, depth=2):
        super(Stage2, self).__init__()
        filter_size = [width*(2**i) for i in range(depth)]
        self.parallel_layers = [ParallelBlock(filter_size[i]) for i in range(depth)]
        self.exchange = ExchangeBlock(layers=2)

    def call(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            output = self.parallel_layers[i](inputs[i])
            outputs.append(output)
        outputs = self.exchange(outputs)
        return outputs


class Stage3(keras.layers.Layer):
    def __init__(self, width=18, depth=3):
        super(Stage3, self).__init__()
        filter_size = [width*(2**i) for i in range(depth)]
        self.parallel_layers = [ParallelBlock(filter_size[i]) for i in range(depth)]
        self.exchange = ExchangeBlock(layers=3)

    def call(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            output = self.parallel_layers[i](inputs[i])
            outputs.append(output)
        outputs = self.exchange(outputs)
        return outputs


class Stage4(keras.layers.Layer):
    def __init__(self, width=18, depth=4):
        super(Stage4, self).__init__()
        filter_size = [width*(2**i) for i in range(depth)]
        self.parallel_layers = [ParallelBlock(filter_size[i]) for i in range(depth)]
        self.exchange = ExchangeBlock(layers=4, within=True)

    def call(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            output = self.parallel_layers[i](inputs[i])
            outputs.append(output)
        outputs = self.exchange(outputs)
        return outputs


class MergeStage_Keypoint(keras.layers.Layer):
    def __init__(self):
        super(MergeStage_Keypoint, self).__init__()

    def call(self, inputs):
        _, x, y, _ = inputs[0].get_shape()
        output = tf.concat([inputs[i] if i==0 else tf.image.resize(inputs[i], [x, y]) for i in range(len(inputs))], 3)
        return output


class HRNet_KeypointHead(keras.Model):
    def __init__(self, num_features, pooling_size=270):
        super(HRNet_KeypointHead, self).__init__()
        self.merge = MergeStage_Keypoint()
        self.last_conv = keras.layers.Conv2D(filters=pooling_size, kernel_size=1, strides=1, padding='same')
        self.last_norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.final_conv = keras.layers.Conv2D(filters=num_features, kernel_size=1, strides=1, padding='same')

    def call(self, inputs, training=False, **kwargs):
        output = self.merge(inputs)
        output = self.last_conv(output)
        output = self.last_norm(output)
        output = self.relu(output)
        output = self.final_conv(output)
        return output


class HRNet_Keypoint(keras.Model):
    def __init__(self, num_features, pooling_size=270):
        super(HRNet_Keypoint, self).__init__()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()

        # Regression Block
        self.merge = MergeStage_Keypoint()
        self.last_conv = keras.layers.Conv2D(filters=pooling_size, kernel_size=1, strides=1, padding='same')
        self.last_norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.final_conv = keras.layers.Conv2D(filters=num_features, kernel_size=1, strides=1, padding='same')

    def call(self, inputs, training=False, **kwargs):
        outputs = self.stage1(inputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)
        output = self.merge(outputs)
        output = self.last_conv(output)
        output = self.last_norm(output)
        output = self.relu(output)
        output = self.final_conv(output)
        return output

    def summary(self, nested=False):
        x = keras.Input(shape=(256, 256, 3))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        if nested:
            return model.summary(expand_nested=True)
        return model.summary()


class MergeStage_Imagenet(keras.layers.Layer):
    def __init__(self, filter_size=128, depth=4):    # [32,64,128,256] OR [128, 256, 512, 1024]
        super(MergeStage_Imagenet, self).__init__()
        filter_size = [filter_size*(2**i) for i in range(4)]
        self.bottleneck_layers = [BottleNeckBlock(out_channel=filter_size[i], trans=True) for i in range(depth)]
        self.conv_layers = []
        for i in range(depth-1):
            block = keras.Sequential([keras.layers.Conv2D(filters=filter_size[i+1], kernel_size=3, strides=2, padding='same'),
                                      keras.layers.BatchNormalization(),
                                      keras.layers.ReLU()])
            self.conv_layers.append(block)

    def call(self, inputs):
        outputs = [self.bottleneck_layers[i](inputs[i]) for i in range(len(inputs))]

        output = outputs[0]
        for i in range(len(outputs)-1):
            output = self.conv_layers[i](output)
            output = tf.add(output, outputs[i+1])
        return output


class HRNet_Imagenet(keras.Model):
    def __init__(self, num_features=1000, pooling_size=270):
        super(HRNet_Imagenet, self).__init__()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()
        self.merge = MergeStage_Imagenet()

        #Classification Block
        self.last_conv = keras.layers.Conv2D(filters=2048, kernel_size=1, strides=1, padding='same')
        self.last_norm = keras.layers.BatchNormalization()
        self.pool = keras.layers.GlobalAveragePooling2D()
        self.fcl = keras.layers.Dense(num_features)
        self.head_block = keras.Sequential([self.last_conv, self.last_norm, keras.layers.ReLU(),
                                            self.pool, self.fcl])

    def call(self, inputs, training=False, **kwargs):
        outputs = self.stage1(inputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)
        output = self.merge(outputs)
        output = self.head_block(output)
        return output

    def summary(self, nested=False):
        x = keras.Input(shape=(256, 256, 3))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        if nested:
            return model.summary(expand_nested=True)
        return model.summary()


class TestLayer(keras.layers.Layer):
    def __init__(self):
        super(TestLayer, self).__init__()
        self.conv = keras.layers.Conv2D(filters=68, kernel_size=1, strides=4, padding='same')
        self.conv_block = keras.Sequential([self.conv])

    def call(self, inputs):
        output = self.conv_block(inputs)
        return output


class TestNet(keras.Model):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layer = TestLayer()

    def call(self, inputs):
        output = self.layer(inputs)
        return output

