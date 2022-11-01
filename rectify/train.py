import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
import load_data
import model as mdl
import heatmap
from hrnet import HRNet_Keypoint, TestNet
from util import load_imagenet, load_aflw, get_transfer_head, get_optimizer


def train(load_trained=False, model='hrnet', dataset='aflw', transfer=False, summary=False, verbose=0):
    assert load_trained or not transfer, f'Pretrained model needed for transfer learning'
    assert model in ['resnet', 'hrnet', 'hrnet_imagenet', 'test'], f'ERROR: Invalid Model: {model}'
    assert dataset in ['aflw', 'imagenet', 'dry'], f'ERROR: Invalid Dataset: {dataset}'
    assert cfg.LOSS_FUNCTION in ['categorical', 'mse'], f'ERROR: Invalid Loss Function: {cfg.LOSS_FUNCTION}'

    # Configs
    checkpoint_path = cfg.MODEL_PATH_CKPT+'.h5py'
    model_path = cfg.MODEL_PATH+'.h5py'
    img_w, img_h, _ = cfg.INPUT_SHAPE
    epoch = cfg.NUM_EPOCH

    if verbose > 0:
        print("Loading dataset...")

    # Fetch dataset
    if dataset == 'aflw':
        training_dataset, validation_dataset = load_aflw(img_w, img_h, return_score=cfg.AFLW_SCORE, use_small=cfg.AFLW_SMALL)
    elif dataset == 'imagenet':
        training_dataset, validation_dataset = load_imagenet(img_w, img_h, val_size=cfg.IMAGENET_VAL_SIZE, normalize=cfg.IMAGENET_NORMALIZE_CPU)
        if verbose > 1:
            plt.figure(figsize=(10, 10))
            for images, labels in training_dataset.take(1):
                for i in range(9):
                    plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i])
                    plt.axis("off")
                plt.show()
    else:
        pass

    if verbose > 0:
        print("Loaded Dataset")

    if verbose > 0:
        print("Loading Model...")

    # Load trained model if needed
    if load_trained:
        model = keras.load_model(model_path)
    elif model == 'resnet':
        model = mdl.create_resnet(cfg.INPUT_SHAPE, 68)
    elif model == 'hrnet':
        model = mdl.create_hrnet(cfg.INPUT_SHAPE, 68)
    elif model == 'hrnet_imagenet':
        if cfg.IMAGENET_NORMALIZE_CPU:
            model = mdl.create_hrnet_imagenet(cfg.INPUT_SHAPE, 1000)
        else:
            model = mdl.create_hrnet_imagenet_with_preprocess(cfg.INPUT_SHAPE, 1000)
    elif model == 'test':
        model = TestNet()

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

    if verbose > 0:
        print("Loaded Model")

    if summary:
        if cfg.TRAIN_PRINT_MODEL_NESTED:
            print(model.summary(nested=cfg.TRAIN_PRINT_MODEL_NESTED))
        else:
            print(model.summary())

    optim, lr_scheduler = get_optimizer()

    # Setup callbacks
    callbacks = [
        lr_scheduler,
        keras.callbacks.EarlyStopping(monitor='loss', patience=7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True,
            verbose=1)
    ]

    if cfg.LOSS_FUNCTION == 'categorical':
        loss = keras.losses.CategoricalCrossentropy()
    elif cfg.LOSS_FUNCTION == 'mse':
        loss = keras.losses.MeanSquaredError()

    # Compile the model
    model.compile(optim,
                  loss,
                  metrics=['accuracy'])
    model.trainable = True

    # Train the model and save
    with tf.device(f'/gpu:{cfg.GPU_NUM}'):
        model.fit(training_dataset, validation_data=validation_dataset, epochs=epoch, callbacks=callbacks, batch_size=cfg.BATCH_SIZE)
        model.save(model_path)


if __name__ == "__main__":
    train(model=cfg.TRAIN_MODEL, dataset=cfg.TRAIN_DATASET, summary=cfg.TRAIN_PRINT_MODEL, verbose=1)
