import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras

import config as cfg
import load_data
import model as mdl


def load_aflw(width, height):
    # Load labeled dataset
    t_x, t_y, v_x, v_y = load_data.load_aflw_train()

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

    return t_ds, v_ds


def get_transfer_head(model_output, num_features, model='resnet'):
    x = keras.layers.GlobalAveragePooling2D()(model_output)
    x = keras.layers.Dense(num_features)(x)
    return x


def train(load_trained=False, model='resnet', dataset='aflw', transfer=False):
    assert load_trained or not transfer, f'Pretrained model needed for transfer learning'
    assert model in ['resnet'], f'ERROR: Invalid Model: {model}'
    assert dataset in ['aflw'], f'ERROR: Invalid Dataset: {dataset}'

    # Configs
    model_path = cfg.MODEL_PATH
    img_w, img_h, _ = cfg.INPUT_SHAPE
    epoch = cfg.NUM_EPOCH
    batch_size = cfg.BATCH_SIZE

    # Fetch dataset
    if dataset=='aflw':
        training_dataset, validation_dataset = load_aflw(img_w, img_h)

    # Load trained model if needed
    if load_trained:
        model = keras.load_model(model_path)
    elif model == 'resnet':
        model = mdl.create_resnet(cfg.INPUT_SHAPE, 18)

    # Freeze layers if transfer learning is enabled
    if transfer:
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
        transfer_head = get_transfer_head(model.output)
        model = keras.models.Model(model.inputs, transfer_head.output)
    else:
        model.trainable = True
        for layer in model.layers:
            layer.trainable = True

    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True,
            verbose=1)
    ]

    # Set learning schedule & optimizer
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    optim = keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optim,
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy', 'mse'])
    model.trainable = True

    # Train the model and save
    with tf.device('/gpu:0'):
        model.fit(training_dataset, validation_data=validation_dataset, epochs=epoch, batch_size=batch_size, callbacks=callbacks)
        model.save(model_path+"model.h5py")


if __name__ == "__main__":
    train()
