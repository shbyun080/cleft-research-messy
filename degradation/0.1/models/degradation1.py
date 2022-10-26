from tensorflow import keras


class Degradation1(keras.Model):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(136, activation='softmax')(x)
        return x

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
