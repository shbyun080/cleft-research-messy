import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras

from hrnet import HRNet_Keypoint, HRNet_Imagenet
import config as cfg


def create_resnet(img_shape=(256, 256, 3), num_features=18):
    resnet = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=img_shape)
    for i in range(len(resnet.layers)):
        resnet.layers[i].trainable = False

    x = keras.layers.GlobalAveragePooling2D()(resnet.output)
    x = keras.layers.Dense(num_features, activation='linear')(x)
    return keras.models.Model(resnet.inputs, x)


def create_hrnet(img_shape=(256, 256, 3), num_features=68):
    hrnet = HRNet_Keypoint(num_features=num_features)
    return hrnet


def create_hrnet_imagenet(img_shape=(256, 256, 3), num_features=1000):
    hrnet = HRNet_Imagenet(num_features=num_features)
    return hrnet


def create_hrnet_imagenet_with_preprocess(img_shape=(256, 256, 3), num_features=1000):
    input = keras.Input(shape=img_shape)
    scale = keras.layers.Rescaling(1./255)
    hrnet = HRNet_Imagenet(num_features=num_features)
    return keras.Sequential([input, scale, hrnet])
