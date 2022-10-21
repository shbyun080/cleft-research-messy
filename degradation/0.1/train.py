import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
import load_data
import degradation2
import numpy as np


def train(model: keras.Model, x, y, training=True):
    checkpoint_filepath = '/saved_weights/run1'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=2),
        model_checkpoint_callback
    ]

    model.trainable = training

    train_x, val_x = x
    train_y, val_y = y

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', 'mse'])
    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=300, batch_size=2, callbacks=callbacks)


if __name__ == "__main__":
    train_cleft = False

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
    t_y = tf.reshape(t_y, [t_y.shape[0], t_y.shape[1] * t_y.shape[2]])
    v_y = tf.reshape(v_y, [v_y.shape[0], v_y.shape[1] * v_y.shape[2]])

    t_x = tf.cast(t_x, tf.float32) / 255.0
    v_x = tf.cast(v_x, tf.float32) / 255.0

    training_model = degradation2.DegradationDeepHRNet()
    training_model.plot_model()
    # print(training_model.summary())
    # train(training_model, (t_x, v_x), (t_y, v_y))
