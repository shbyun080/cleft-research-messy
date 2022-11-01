import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

import config as cfg


def load_cleft_train():
    """
    Returns
    _______
    train_x, train_y, valid_x, valid_y
        a tuple containing training and validation sets
    """
    train_x = np.load("../data/cleft/train_x.npy") / 255
    train_y = np.load("../data/cleft/train_y.npy")  # /dim

    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

    return train_x, train_y, valid_x, valid_y


def load_cleft_test():
    """
    Returns
    _______
    test_x, test_y
        a tuple containing test sets
    """
    test_x = np.load("../data/cleft/test_x.npy") / 255
    test_y = np.load("../data/cleft/test_y.npy")  # /dim

    return test_x, test_y


def load_300w_train():
    """
    Returns
    _______
    train_x, train_y, valid_x, valid_y
        a tuple containing training and validation sets
    """
    data, info = tfds.load('the300w_lp', split='train', shuffle_files=True, with_info=True, data_dir='../../data/300w')
    print("splitting...")
    train_x, valid_x, train_y, valid_y = train_test_split(data['image'], data['landmarks_2d'], test_size=0.1)
    print("splitting complete")

    return train_x, train_y, valid_x, valid_y


def load_aflw_train():
    """
    Returns
    _______
    train_x, train_y, valid_x, valid_y
        a tuple containing training and validation sets
    """
    train_data, train_info = tfds.load('aflw2k3d', split='train[:90%]', with_info=True, data_dir='../data/aflw',
                                       batch_size=-1)
    valid_data, valid_info = tfds.load('aflw2k3d', split='train[90%:]', with_info=True, data_dir='../data/aflw',
                                       batch_size=-1)

    train_x, train_y = train_data['image'], train_data['landmarks_68_3d_xy_normalized']
    valid_x, valid_y = valid_data['image'], valid_data['landmarks_68_3d_xy_normalized']

    return train_x, train_y, valid_x, valid_y


def load_imagenet_train(width, height, val_size=10):
    t_ds = keras.preprocessing.image_dataset_from_directory(
        cfg.IMAGENET_TRAIN_DIR,
        label_mode="categorical",
        color_mode="rgb",
        batch_size=cfg.BATCH_SIZE,
        image_size=(256, 256),
        shuffle=True,
        seed=cfg.IMAGENET_SEED,
        validation_split=val_size/100,
        subset='training'
    )

    v_ds = keras.preprocessing.image_dataset_from_directory(
        cfg.IMAGENET_TRAIN_DIR,
        label_mode="categorical",
        color_mode="rgb",
        batch_size=cfg.BATCH_SIZE,
        image_size=(width, height),
        shuffle=True,
        seed=cfg.IMAGENET_SEED,
        validation_split=val_size/100,
        subset='validation'
    )

    return t_ds, v_ds


def load_imagenet_train_archaic(subset=1):
    """
    Returns
    _______
    train_x, train_y, valid_x, valid_y
        a tuple containing training and validation sets
    """
    # TODO Download, Setup, Load Imagenet2012 Dataset
    # Get imagenet labels
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    # Set data_dir to a read-only storage of .tar files
    # Set write_dir to a w/r storage
    data_dir = '../data/imagenet/data'
    write_dir = '../data/imagenet/store/tf-imagenet-dirs'

    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
        extract_dir=os.path.join(write_dir, 'extracted'),
        manual_dir=data_dir
    )
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'downloaded'),
        'download_config': download_config,
    }
    train_ds, train_info = tfds.load('imagenet2012_subset',
                                     data_dir=os.path.join(write_dir, 'data'),
                                     split=f'train[:{subset}%]',
                                     with_info=True,
                                     shuffle_files=False,
                                     download=True,
                                     as_supervised=True,
                                     download_and_prepare_kwargs=download_and_prepare_kwargs,
                                     batch_size=-1)

    valid_ds, valid_info = tfds.load('imagenet2012_subset',
                                     data_dir=os.path.join(write_dir, 'data'),
                                     split=f'validation[:{subset}%]',
                                     with_info=True,
                                     shuffle_files=False,
                                     download=True,
                                     as_supervised=True,
                                     download_and_prepare_kwargs=download_and_prepare_kwargs,
                                     batch_size=-1)

    # return train_ds, valid_ds

    train_x = train_ds[0]
    train_y = train_ds[1]
    valid_x = valid_ds[0]
    valid_y = valid_ds[1]

    # train_y = tf.one_hot(train_y, 1000)
    # valid_y = tf.one_hot(valid_y, 1000)

    train_y = keras.utils.to_categorical(train_y, num_classes=1000)
    valid_y = keras.utils.to_categorical(valid_y, num_classes=1000)

    return train_x, train_y, valid_x, valid_y
