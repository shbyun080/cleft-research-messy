import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
import numpy as np

import config as cfg
import load_data
import heatmap


def load_aflw(width, height, return_score=True, use_small=False):
    # Load labeled dataset
    t_x, t_y, v_x, v_y = load_data.load_aflw_train()

    if use_small:
        t_x = t_x[:32]
        t_y = t_y[:32]
        v_x = v_x[:32]
        v_y = v_y[:32]

    if return_score:
        # Generate heatmap
        t_y = heatmap.to_heatmap(t_y)
        v_y = heatmap.to_heatmap(v_y)
        t_y = np.moveaxis(t_y, 1, -1)
        v_y = np.moveaxis(v_y, 1, -1)
    else:
        # Flatten labels
        t_y = tf.reshape(t_y, [t_y.shape[0], -1])
        v_y = tf.reshape(v_y, [v_y.shape[0], -1])

    # Resize images
    t_x = tf.image.resize(t_x, [width, height])
    v_x = tf.image.resize(v_x, [width, height])

    # Normalize images
    t_x = tf.cast(t_x, tf.float32) / 255.0
    v_x = tf.cast(v_x, tf.float32) / 255.0

    # Generate dataset from images and labels
    t_ds = tf.data.Dataset.from_tensor_slices((t_x, t_y))
    v_ds = tf.data.Dataset.from_tensor_slices((v_x, v_y))

    t_ds = t_ds.batch(batch_size=cfg.BATCH_SIZE)
    v_ds = v_ds.batch(batch_size=cfg.BATCH_SIZE)

    return t_ds, v_ds


def load_imagenet(width, height, val_size=10, normalize=True):
    t_ds, v_ds = load_data.load_imagenet_train(width, height, val_size)

    if normalize:
        t_ds = t_ds.map(imagenet_normalize)
        v_ds = v_ds.map(imagenet_normalize)

    t_ds = t_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    v_ds = v_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return t_ds, v_ds


def imagenet_normalize(image, label):
    image = tf.cast(image/255., tf.float32)
    label = tf.cast(label, tf.float32)

    return image, label


def load_imagenet_archaic(width, height, subset=1):
    # Load labeled dataset
    # t_ds, v_ds = load_data.load_imagenet_train(subset)
    t_x, t_y, v_x, v_y = load_data.load_imagenet_train(subset)


    # t_ds = t_ds.map(imagenet_preprocess)
    # v_ds = v_ds.map(imagenet_preprocess)

    # Resize images
    t_x = tf.image.resize(t_x, [width, height])
    v_x = tf.image.resize(v_x, [width, height])

    # Normalize images
    t_x = tf.cast(t_x, tf.float32) / 255.0
    v_x = tf.cast(v_x, tf.float32) / 255.0

    # Generate dataset from images and labels
    t_ds = tf.data.Dataset.from_tensor_slices((t_x, t_y))
    v_ds = tf.data.Dataset.from_tensor_slices((v_x, v_y))

    t_ds = t_ds.batch(batch_size=cfg.BATCH_SIZE)
    v_ds = v_ds.batch(batch_size=cfg.BATCH_SIZE)

    return t_ds, v_ds


def imagenet_preprocess(image, label):
    x, y, _ = cfg.INPUT_SHAPE
    i = image
    i = tf.cast(i, tf.float32)
    i = i / 255.0
    i = tf.image.resize(i, [x, y])
    return i, label


def get_optimizer():
    assert cfg.OPTIMIZER in ['sgd', 'adam'], f'Invalid Optimizer: {cfg.OPTIMIZER}'

    lr_scheduler = None

    if cfg.OPTIMIZER == 'sgd':

        def scheduler(epoch, lr):
            if epoch in [30, 60, 90]:
                return lr * 0.1
            return lr

        optimizer = keras.optimizers.SGD(
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            decay=0.0001
        )
        lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001
        )

    return optimizer, lr_scheduler


def get_transfer_head(model_output, num_features, model='resnet'):
    x = keras.layers.GlobalAveragePooling2D()(model_output)
    x = keras.layers.Dense(num_features)(x)
    return x