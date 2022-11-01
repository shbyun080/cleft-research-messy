import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
import load_data
import degradation2
from models import degradation1
import numpy as np
from degredationHRNet import HRNet_Keypoint, DegradeNet
from test_model import Net
from utils import HRNetLearningRate
from resnet import create_resnet
from heatmap import to_heatmap


def train(model: keras.Model, x=None, y=None, training=True, t_ds=None, v_ds=None, continue_train=False):
    checkpoint_filepath = './saved_weights/hrnet/10_26_2022_6am/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
        verbose=1)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=7, verbose=1),
        model_checkpoint_callback
    ]

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.92)

    optim = keras.optimizers.Adam(learning_rate=lr_schedule)

    if continue_train:
        model.load_weights(checkpoint_filepath+"model.h5py")

    model.compile(optim,
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy', 'mse'])

    model.trainable = training
    for layer in model.layers:
        layer.trainable = True

    with tf.device('/gpu:1'):
        if t_ds==None:
            print("None-Dataset Training")
            train_x, val_x = x
            train_y, val_y = y
            model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=500, batch_size=2, callbacks=callbacks)
        else:
            print("Dataset Training")
            model.fit(t_ds, validation_data=v_ds, epochs=2, batch_size=16, callbacks=callbacks)
        model.save_weights(checkpoint_filepath+"model.h5py")


if __name__ == "__main__":
    train_cleft = False
    preloaded_heatmap = False

    if not preloaded_heatmap:
        if train_cleft:
            t_x, t_y, v_x, v_y = load_data.load_cleft_train()  # LOAD CLEFT IMAGE
        else:
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

        t_x = t_x[:32]
        t_y = t_y[:32]
        v_x = v_x[:32]
        v_y = v_y[:32]

        # Heatmap Generation
        t_y = to_heatmap(t_y, verbose = 1)
        v_y = to_heatmap(v_y, verbose = 1)

        t_y = np.moveaxis(t_y, 1, -1)
        v_y = np.moveaxis(v_y, 1, -1)

        t_x = tf.image.resize(t_x, [256, 256])
        v_x = tf.image.resize(v_x, [256, 256])

        t_x = tf.cast(t_x, tf.float32) / 255.0
        v_x = tf.cast(v_x, tf.float32) / 255.0

        t_ds = tf.data.Dataset.from_tensor_slices((t_x, t_y))
        v_ds = tf.data.Dataset.from_tensor_slices((v_x, v_y))

        # tf.data.Dataset.save(t_ds, "./saved_dataset/train")
        # tf.data.Dataset.save(v_ds, "./saved_dataset/validate")
    else:
        t_ds = tf.data.Dataset.load("./saved_dataset/train")
        v_ds = tf.data.Dataset.load("./saved_dataset/validate")

    t_ds = t_ds.batch(batch_size=1)
    v_ds = v_ds.batch(batch_size=1)

    # training_model = create_resnet()
    # training_model = HRNet_Keypoint(num_features=68)
    # training_model = DegradeNet()
    training_model = Net()
    # training_model = degradation2.DegradationDeepHRNet()
    # training_model.plot_model()
    print(training_model.summary(nested=False))
    # train(training_model, x=(t_x, v_x), y=(t_y, v_y))
    # train(training_model, t_ds=t_ds, v_ds=v_ds, continue_train=False)
