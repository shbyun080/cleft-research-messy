import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras


class BottleNeckBlock(keras.Model):
    def __init__(self, channel=256, trans=False):
        super(BottleNeckBlock, self).__init__()
        self.relu = keras.layers.ReLU()
        self.norm1 = keras.layers.BatchNormalization()
        self.norm2 = keras.layers.BatchNormalization()
        self.norm3 = keras.layers.BatchNormalization()
        self.conv1 = keras.layers.Conv2D(filters=channel, kernel_size=1, strides=1, padding='same')
        self.conv2 = keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')
        self.conv3 = keras.layers.Conv2D(filters=channel, kernel_size=1, strides=1, padding='same')

        self.trans = trans
        if self.trans:
            self.trans_conv = keras.layers.Conv2D(filters=channel, kernel_size=1, strides=1, padding='same')
            self.trans_norm = keras.layers.BatchNormalization()

    def call(self, inputs):
        residual = inputs

        outputs = self.conv1(residual)
        outputs = self.norm1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.norm2(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv3(outputs)
        outputs = self.norm3(outputs)

        if self.trans:
            residual = self.trans_conv(residual)
            residual = self.trans_norm(residual)

        _outputs = tf.add(residual, outputs)
        outputs = self.relu(_outputs)

        return outputs


class ExchangeBlock(keras.Model):
    def __init__(self, layers, within=False, filter_size=18):      # [32,64,128,256] OR [128, 256, 512, 1024]
        super(ExchangeBlock, self).__init__()
        filter_size = [filter_size*(2**i) for i in range(4)]
        self.num_layers = layers
        self.within = within

        self.relu = keras.layers.ReLU()

        self.conv_layers = []
        self.norm_layers = []
        for i in range(layers):
            one_layer = []
            one_norm = []
            for j in range(layers):
                if i == j:
                    one_layer.append(None)
                    one_norm.append(None)
                elif i < j:
                    one_layer.append(
                        keras.layers.Conv2D(filters=filter_size[i], kernel_size=1, strides=1, padding='valid'))
                    one_norm.append(keras.layers.BatchNormalization())
                else:
                    mul_conv_layers = []
                    mul_norm_layers = []
                    for k in range(i-j):
                        conv = keras.layers.Conv2D(filters=filter_size[i], kernel_size=3, strides=2, padding='same')
                        norm = keras.layers.BatchNormalization()
                        mul_conv_layers.append(conv)
                        mul_norm_layers.append(norm)
                    one_layer.append(mul_conv_layers)
                    one_norm.append(mul_norm_layers)
            self.conv_layers.append(one_layer)
            self.norm_layers.append(one_norm)

        if not self.within:
            self.last_conv = keras.layers.Conv2D(filters=filter_size[layers], kernel_size=3, strides=2, padding='same')
            self.last_norm = keras.layers.BatchNormalization()

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
                    temp_net = self.norm_layers[i][j](temp_net)
                    temp_net = tf.image.resize(temp_net, [x, y])
                    temp_net = self.relu(temp_net)
                else:  # i is lower resolution
                    temp_net = inputs[j]
                    for k in range(len(self.conv_layers[i][j])):
                        temp_net = self.conv_layers[i][j][k](temp_net)
                        temp_net = self.norm_layers[i][j][k](temp_net)
                        temp_net = self.relu(temp_net)
                curr_net = tf.add(temp_net, curr_net)
            outputs.append(curr_net)

        if not self.within:
            subnetwork = self.last_conv(outputs[-1])
            subnetwork = self.last_norm(subnetwork)
            subnetwork = self.relu(subnetwork)
            outputs.append(subnetwork)

        return outputs


class ParallelBlock(keras.Model):
    def __init__(self, layers, filter_size=18):
        super(ParallelBlock, self).__init__()
        filter_size = [filter_size*(2**i) for i in range(4)]
        self.num_layers = layers
        self.relu = keras.layers.ReLU()
        # self.pool = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.conv_layers = []
        self.norm_layers = []
        for i in range(layers):
            conv = keras.layers.Conv2D(filters=filter_size[i], kernel_size=3, strides=1, padding='same')
            self.conv_layers.append(conv)
            self.norm_layers.append(keras.layers.BatchNormalization())

    def call(self, inputs):
        outputs = []
        for i in range(self.num_layers):
            output = self.conv_layers[i](inputs[i])
            output = self.norm_layers[i](output)
            output = self.relu(output)
            outputs.append(output)
        return outputs


class Stage1(keras.Model):
    def __init__(self, filter_size=18):
        super(Stage1, self).__init__()
        self.relu = keras.layers.ReLU()
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.batch1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.batch2 = keras.layers.BatchNormalization()
        self.bottleneck_layers = [BottleNeckBlock(channel=64, trans=True),
                                  BottleNeckBlock(channel=64),
                                  BottleNeckBlock(channel=64),
                                  BottleNeckBlock(channel=64)]
        self.conv3 = keras.layers.Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same')
        self.batch3 = keras.layers.BatchNormalization()
        self.exchange = ExchangeBlock(layers=1)

    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.batch1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.batch2(outputs)
        outputs = self.relu(outputs)
        for i in range(len(self.bottleneck_layers)):
            outputs = self.bottleneck_layers[i](outputs)
        outputs = self.conv3(outputs)
        outputs = self.batch3(outputs)
        outputs = self.relu(outputs)
        outputs = [outputs]
        outputs = self.exchange(outputs)
        return outputs


class Stage2(keras.Model):
    def __init__(self):
        super(Stage2, self).__init__()
        self.parallel_layers = [ParallelBlock(layers=2),
                                ParallelBlock(layers=2),
                                ParallelBlock(layers=2),
                                ParallelBlock(layers=2)]
        self.exchange = ExchangeBlock(layers=2)

    def call(self, inputs):
        outputs = inputs
        for i in range(len(self.parallel_layers)):
            outputs = self.parallel_layers[i](outputs)
        outputs = self.exchange(outputs)
        return outputs


class Stage3(keras.Model):
    def __init__(self):
        super(Stage3, self).__init__()
        self.parallel_layers = [ParallelBlock(layers=3),
                                ParallelBlock(layers=3),
                                ParallelBlock(layers=3),
                                ParallelBlock(layers=3)]
        self.exchange = ExchangeBlock(layers=3)

    def call(self, inputs):
        outputs = inputs
        for i in range(len(self.parallel_layers)):
            outputs = self.parallel_layers[i](outputs)
        outputs = self.exchange(outputs)
        return outputs


class Stage4(keras.Model):
    def __init__(self):
        super(Stage4, self).__init__()
        self.parallel_layers = [ParallelBlock(layers=4),
                                ParallelBlock(layers=4),
                                ParallelBlock(layers=4),
                                ParallelBlock(layers=4)]
        self.exchange = ExchangeBlock(layers=4, within=True)

    def call(self, inputs):
        outputs = inputs
        for i in range(len(self.parallel_layers)):
            outputs = self.parallel_layers[i](outputs)
        outputs = self.exchange(outputs)
        return outputs


class MergeStage_Imagenet(keras.Model):
    def __init__(self, filter_size=128):    # [32,64,128,256] OR [128, 256, 512, 1024]
        super(MergeStage_Imagenet, self).__init__()
        filter_size = [filter_size*(2**i) for i in range(4)]
        self.bottleneck_layers = [BottleNeckBlock(channel=filter_size[0], trans=True),
                                  BottleNeckBlock(channel=filter_size[1], trans=True),
                                  BottleNeckBlock(channel=filter_size[2], trans=True),
                                  BottleNeckBlock(channel=filter_size[3], trans=True)]
        self.relu = keras.layers.ReLU()
        self.conv_layers = []
        self.batch_layers = []
        # for i in range(3):
        #     self.conv_layers.append(keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same'))
        #     self.batch_layers.append(keras.layers.BatchNormalization())
        for i in range(3):
            self.conv_layers.append(keras.layers.Conv2D(filters=filter_size[i+1], kernel_size=3, strides=2, padding='same'))
            self.batch_layers.append(keras.layers.BatchNormalization())

    def call(self, inputs):
        outputs = []
        for i in range(len(self.bottleneck_layers)):
            temp = self.bottleneck_layers[i](inputs[i])
            outputs.append(temp)

        output = outputs[0]
        for i in range(len(outputs)-1):
            output = self.conv_layers[i](output)
            output = self.batch_layers[i](output)
            output = self.relu(output)
            output = tf.add(output, outputs[i+1])
        return output


class MergeStage_Keypoint(keras.Model):
    def __init__(self, filter_size=128):    # [32,64,128,256] OR [128, 256, 512, 1024]
        super(MergeStage_Keypoint, self).__init__()
        # filter_size = [filter_size*(2**i) for i in range(4)]
        # self.bottleneck_layers = [BottleNeckBlock(channel=filter_size[0], trans=True),
        #                           BottleNeckBlock(channel=filter_size[1], trans=True),
        #                           BottleNeckBlock(channel=filter_size[2], trans=True),
        #                           BottleNeckBlock(channel=filter_size[3], trans=True)]

    def call(self, inputs):
        outputs = inputs
        # outputs = []
        # for i in range(len(self.bottleneck_layers)):
        #     temp = self.bottleneck_layers[i](inputs[i])
        #     outputs.append(temp)

        _, x, y, _ = outputs[0].get_shape()
        for i in range(1, len(outputs)):
            outputs[i] = tf.image.resize(outputs[i], [x, y])

        output = tf.concat(outputs, 3)

        return output


class HRNetBase(keras.Model):
    def __init__(self):
        super(HRNetBase, self).__init__()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()

    def call(self, inputs):
        outputs = self.stage1(inputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)
        return outputs


class HRNet_Imagenet(keras.Model):
    def __init__(self, num_classes, pooling_size=1920):
        super(HRNet_Imagenet, self).__init__()
        self.base = HRNetBase()
        self.merge = MergeStage_Imagenet()
        self.last_conv = keras.layers.Conv2D(filters=pooling_size, kernel_size=1, strides=1, padding='same')
        self.last_norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.pool = keras.layers.GlobalAveragePooling2D()
        # self.flat = keras.layers.Flatten()
        # self.fcl1 = keras.layers.Dense(256, activation='relu')
        # self.drop1 = keras.layers.Dropout(0.2)
        self.fcl2 = keras.layers.Dense(num_classes, activation='linear')

    def call(self, inputs):
        outputs = self.base(inputs)
        output = self.merge(outputs)
        output = self.last_conv(output)
        output = self.last_norm(output)
        output = self.relu(output)
        output = self.pool(output)
        # output = self.flat(output[0])
        # output = self.fcl1(output)
        # output = self.drop1(output)
        output = self.fcl2(output)
        output = tf.identity(output)
        return output

    def summary(self, nested=False):
        x = keras.Input(shape=(256, 256, 3))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        if nested:
            return model.summary(expand_nested=True)
        return model.summary()


class HRNet_Keypoint(keras.Model):
    def __init__(self, num_features, pooling_size=270):
        super(HRNet_Keypoint, self).__init__()
        self.base = HRNetBase()
        self.merge = MergeStage_Keypoint()
        self.last_conv = keras.layers.Conv2D(filters=pooling_size, kernel_size=1, strides=1, padding='same')
        self.last_norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.final_conv = keras.layers.Conv2D(filters=num_features, kernel_size=3, strides=1, padding='same')

    def call(self, inputs):
        outputs = self.base(inputs)
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


class Degradation(keras.Model):
    def __init__(self):
        super(Degradation, self).__init__()
        self.relu = keras.layers.ReLU()

        self.conv_layers = []
        self.norm_layers = []
        for i in range(3):
            conv = keras.layers.Conv2D(filters=32, kernel_size=2, strides=2, padding='same')
            self.conv_layers.append(conv)
            self.norm_layers.append(keras.layers.BatchNormalization())
        conv = keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same')
        self.conv_layers.append(conv)
        self.norm_layers.append(keras.layers.BatchNormalization())

    def call(self, inputs):
        _, x, y, _ = inputs.get_shape()
        outputs = inputs
        for i in range(len(self.conv_layers)):
            outputs = self.conv_layers[i](outputs)
            outputs = self.norm_layers[i](outputs)
            outputs = self.relu(outputs)

        outputs = tf.image.resize(outputs, [x, y])
        outputs = tf.add(inputs, outputs)
        return outputs


class DegradeNet(keras.Model):
    def __init__(self):
        super(DegradeNet, self).__init__()
        self.degrade = Degradation()
        self.hrnet = HRNet_Imagenet(num_features=136)

    def forward(self, inputs):
        outputs = self.degrade(inputs)
        outputs = self.hrnet(outputs)
        return outputs

    def call(self, inputs):
        return self.forward(inputs)

    def summary(self):
        x = keras.Input(shape=(448, 448, 3))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


if __name__ == "__main__":
    degradenet = DegradeNet()
    print(degradenet.summary())
