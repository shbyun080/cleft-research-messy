import tensorflow as tf
from tensorflow import keras
from models import hrnet_util

import pydotplus
from keras.utils.vis_utils import model_to_dot
import keras
keras.utils.vis_utils.pydot = pydotplus

def degrade_block(x):
    x = hrnet_util.conv_2d(x, channels=32, kernel_size=2, activation=hrnet_util.relu)
    # x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

    residual = x
    for _ in range(4):
        x = hrnet_util.conv_2d(x, channels=32, kernel_size=2, strides=2, activation=hrnet_util.relu)
        # Consider using 1 convolution instead
        # x = keras.layers.Conv2D(x, filters=64, kernel_size=(2, 2), strides=(2, 2), padding='valid')
        # x = keras.layers.BatchNormalization(x)
        # x = keras.layers.ReLU(x)
        # Consider not max-pooling
        # x = keras.layers.MaxPooling2D(x, pool_size=(2, 2), strides=(2, 2), padding='valid')
        x = keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last", interpolation="bilinear")(x)
    x = keras.layers.Add()([residual, x])
    return x


@tf.function
def deep_hrnet(x, num_features, eps=1e-10):
    def stage1(input, name='stage1'):
        outputs = []
        s1_res1 = hrnet_util.residual_unit_bottleneck(input)
        s1_res2 = hrnet_util.residual_unit_bottleneck(s1_res1)
        s1_res3 = hrnet_util.residual_unit_bottleneck(s1_res2)
        s1_res4 = hrnet_util.residual_unit_bottleneck(s1_res3)
        outputs.append(hrnet_util.conv_2d(s1_res4, channels=32, activation=hrnet_util.relu))
        return outputs

    def stage2(input, name='stage2'):
        sub_networks = hrnet_util.exchange_between_stage(input)
        sub_networks = hrnet_util.exchange_block(sub_networks)
        return sub_networks

    def stage3(input, name='stage3'):
        sub_networks = hrnet_util.exchange_between_stage(input)
        sub_networks = hrnet_util.exchange_block(sub_networks)
        sub_networks = hrnet_util.exchange_block(sub_networks)
        sub_networks = hrnet_util.exchange_block(sub_networks)
        sub_networks = hrnet_util.exchange_block(sub_networks)
        return sub_networks

    def stage4(input, name='stage4'):
        sub_networks = hrnet_util.exchange_between_stage(input)
        sub_networks = hrnet_util.exchange_block(sub_networks)
        sub_networks = hrnet_util.exchange_block(sub_networks)
        sub_networks = hrnet_util.exchange_block(sub_networks)
        return sub_networks

    output = stage1(input=x)
    output = stage2(input=output)
    output = stage3(input=output)
    # output = stage4(input=output)

    # The output contains 4 subnetworks, we only need the first one
    output = output[0]

    # using a 3x3 convolution to reduce the channels of feature maps to 14 (the number of keypoints)
    # output = hrnet_util.conv_2d(output, channels=num_features, kernel_size=1, strides=1, batch_normalization=False, activation=hrnet_util.relu)

    # In order to avoid this from happening, we need to normalize the value of the net output by dividing the value on
    # all pixels by the sum of the value on that image (1, 256, 192, 1). Or we may calculate the classification loss
    # to indicate the class of the key points.

    # sum up the value on each pixels, the result should be a [batch_size, 14] tensor, then expend dim to be
    # [batch_size, 1, 1, num_features] tensor to normalize the output
    # output_sum = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.reduce_sum(output, axis=-2), axis=-2), axis=-2),
    #                             axis=-2)
    #
    # output = tf.truediv(output, output_sum + eps)
    # output = keras.layers.GlobalAveragePooling2D()(output)

    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(136, activation='softmax')(output)
    output = keras.layers.Flatten()(output)

    return output


class DegradationDeepHRNet(keras.Model):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = degrade_block(x)
        x = deep_hrnet(x, num_features=68 * 2)  # 68*2 for AFLW
        return x

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)

    def summary(self):
        x = keras.Input(shape=(224, 224, 3))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def plot_model(self):
        x = keras.Input(shape=(224, 224, 3))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        keras.utils.vis_utils.plot_model(
            model,
            to_file="model.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )