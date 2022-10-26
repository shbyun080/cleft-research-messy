import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
from degredationHRNet import HRNet_Keypoint
import load_data
import numpy as np
import matplotlib.pyplot as plt
from heatmap import to_heatmap, decode_preds

def test(load_path, test_image, test_label):
    # model = keras.models.load_model(load_path+"model.h5")
    # model.trainable = False
    # model.compile(keras.optimizers.Adam(),
    #                             loss=keras.losses.MeanSquaredError(),
    #                             metrics=['accuracy', 'mse'])
    # print(model.evaluate(test_image, test_label[:, 54:72], batch_size=16, verbose=2))

    model = HRNet_Keypoint(num_features=68)
    model.compile(keras.optimizers.Adam(),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy', 'mse'])
    model.load_weights(load_path)
    model.trainable = False
    score = model.predict(test_image)

    score = np.moveaxis(score, -1, 1)

    f = plt.figure(0)
    plt.imshow(test_image[0])

    for s in score[0]:
        print_heatmap()

    plt.show()

    # pts = decode_preds(score)
    #
    # plt.imshow(test_image[1])
    # plt_pts = pts[1]
    # # plt_pts = np.reshape(test_label[0][54:72], (9,2))
    # for x, y in plt_pts:
    #     plt.plot(x*256,y*256,'ro', markersize=0.5)
    # plt.show()


def print_heatmap(score, id=0):
    f = plt.figure(id)
    plt.imshow(score)
    f.show()


if __name__=='__main__':
    testing_path = "./saved_weights/hrnet/10_26_2022_6am/model.h5py"

    t_x, t_y, v_x, v_y = load_data.load_aflw_train()  # LOAD TENSORFLOW AFLW DATASET

    # TEST AN IMAGE TO CHECK VALIDITY
    # print(t_x.shape, t_y.shape, v_x.shape, v_y.shape)
    # plt.imshow(t_x[365,:,:,:])
    # for x,y in t_y[365,:,:]:
    #     plt.plot(x*450,y*450,'ro', markersize=0.5)
    # plt.show()

    # Flatten x,y for training
    # t_y = tf.reshape(t_y, [t_y.shape[0], t_y.shape[1] * t_y.shape[2]])
    # v_y = tf.reshape(v_y, [v_y.shape[0], v_y.shape[1] * v_y.shape[2]])

    v_x = v_x[0:2]
    v_y = v_y[0:2]

    print(v_y[0])

    # Heatmap Generation
    v_y = to_heatmap(v_y)
    v_y = np.moveaxis(v_y, 1, -1)

    v_x = tf.image.resize(v_x, [256, 256])
    v_x = tf.cast(v_x, tf.float32) / 255.0

    test_images = v_x
    test_labels = v_y

    test(testing_path, test_images, test_labels)