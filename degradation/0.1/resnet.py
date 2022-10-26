from tensorflow import keras


def create_resnet():
    resnet = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
    for i in range(len(resnet.layers)):
        resnet.layers[i].trainable = False

    x = keras.layers.GlobalAveragePooling2D()(resnet.output)
    x = keras.layers.Dense(18, activation='linear')(x)
    return keras.models.Model(resnet.inputs, x)

