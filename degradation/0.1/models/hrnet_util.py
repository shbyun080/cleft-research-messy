import tensorflow as tf
from tensorflow import keras


def relu(inputs):
    return keras.layers.ReLU()(inputs)


def conv_2d(inputs, channels, kernel_size=3, strides=1, batch_normalization=True, activation=None, padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)):
    output = keras.layers.Conv2D(filters=channels, kernel_size=kernel_size, strides=strides,
                                 padding=padding, kernel_initializer=kernel_initializer)(inputs)

    if batch_normalization:
        output = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(output)

    if activation:
        output = activation(output)

    return output


def down_sampling(input, method='strided_convolution', rate=2, name='', activation=relu):
    if method == 'strided_convolution':
        _, _, _, channels = input.get_shape()
        output = input
        loop_index = 1
        new_rate = rate  # power of 2
        while new_rate > 1:
            output = conv_2d(output, channels=channels * (2 ** loop_index), strides=2, activation=activation)
            loop_index += 1
            new_rate = int(new_rate / 2)

    elif method == 'max_pooling':
        output = keras.layers.MaxPool2D(input, pool_size=rate, strides=rate, name=name + '_max_pooling')

    return output


def up_sampling(input, channels, rate=2, name='', activation=relu):
    _, x, y, _ = input.get_shape()

    output = tf.image.resize(input, [x * rate, y * rate])
    output = conv_2d(output, channels=channels, kernel_size=1, activation=activation)

    return output


# Repeated multiscale fusion (namely the exchange block) within a stage (the input and the output has the same
# number of subnetworks)
def exchange_within_stage(inputs):
    subnetworks_number = len(inputs)
    outputs = []

    # suppose i is the index of the input subnetwork, o is the index of the output subnetwork
    # higher subnetwork index indicates higher resolution
    for o in range(subnetworks_number):
        one_subnetwork = 0
        for i in range(subnetworks_number):
            if i == o:
                # if in the same resolution
                temp_subnetwork = inputs[i]
            elif i - o < 0:
                # if the input resolution is greater, down-sampling with rate of 2 ** (o - i)
                temp_subnetwork = down_sampling(inputs[i], rate=2 ** (o - i))
            else:
                # if the input resolution is smaller, up-sampling with rate of 2 ** (o - i)
                c = inputs[o].get_shape()[3]
                temp_subnetwork = up_sampling(inputs[i], channels=c, rate=2 ** (i - o))
            one_subnetwork = tf.add(temp_subnetwork, one_subnetwork)
        outputs.append(one_subnetwork)
    return outputs


# Repeated multiscale fusion (namely the exchange block) between two stages (the input and the output has the same
# number of subnetworks)
def exchange_between_stage(inputs):
    outputs = exchange_within_stage(inputs)
    one_subnetwork = down_sampling(inputs[-1], rate=2)
    outputs.append(one_subnetwork)
    return outputs


def residual_unit_bottleneck(input, channels=64):
    """
    Residual unit with bottleneck design, default width is 64.
    """
    c = input.get_shape()[3]  # Number of channels
    conv_1x1_1 = conv_2d(input, channels=channels, kernel_size=1, activation=relu)
    conv_3x3 = conv_2d(conv_1x1_1, channels=channels, activation=relu)
    conv_1x1_2 = conv_2d(conv_3x3, channels=c, kernel_size=1)
    _output = tf.add(input, conv_1x1_2)
    output = relu(_output)
    return output


def residual_unit(input):
    """
    Residual unit with two 3 x 3 convolution layers.
    """
    c = input.get_shape()[3]  # Number of channels
    conv3x3_1 = conv_2d(inputs=input, channels=c, activation=relu)
    conv3x3_2 = conv_2d(inputs=conv3x3_1, channels=c)
    _output = tf.add(input, conv3x3_2)
    output = relu(_output)
    return output


def exchange_block(inputs):
    output = []
    level = 0
    for input in inputs:
        sub_network = residual_unit(input)
        sub_network = residual_unit(sub_network)
        sub_network = residual_unit(sub_network)
        sub_network = residual_unit(sub_network)
        output.append(sub_network)
        level += 1
    outputs = exchange_within_stage(output)
    return outputs
