import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds


def load_cleft_train():
    """
    Returns
    _______
    train_x, train_y, valid_x, valid_y
        a tuple containing training and validation sets
    """
    train_x = np.load("../../data/cleft/train_x.npy") / 255
    train_y = np.load("../../data/cleft/train_y.npy")  # /dim

    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

    return train_x, train_y, valid_x, valid_y


def load_cleft_test():
    """
    Returns
    _______
    test_x, test_y
        a tuple containing test sets
    """
    test_x = np.load("../../data/cleft/test_x.npy") / 255
    test_y = np.load("../../data/cleft/test_y.npy")  # /dim

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


def load_300w_test():
    """
    Returns
    _______
    test_x, test_y
        a tuple containing test sets
    """
    data, info = tfds.load('the300w_lp_test', split='train', shuffle_files=False, with_info=True)

    return data['image'], data['landmarks_2d']


def load_aflw_train():
    """
    Returns
    _______
    train_x, train_y, valid_x, valid_y
        a tuple containing training and validation sets
    """
    train_data, train_info = tfds.load('aflw2k3d', split='train[:90%]', with_info=True, data_dir='../../data/aflw',
                                       batch_size=-1)
    valid_data, valid_info = tfds.load('aflw2k3d', split='train[90%:]', with_info=True, data_dir='../../data/aflw',
                                       batch_size=-1)

    # train_data = tfds.as_numpy(train_data)
    # valid_data = tfds.as_numpy(valid_data)

    train_x, train_y = train_data['image'], train_data['landmarks_68_3d_xy_normalized']
    valid_x, valid_y = valid_data['image'], valid_data['landmarks_68_3d_xy_normalized']

    return train_x, train_y, valid_x, valid_y
