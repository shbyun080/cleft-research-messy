from tensorflow import keras


class Degradation1(keras.Model):
    def __init__(self):
        super().__init__()

    def init_layer(self, x):
        x = keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = keras.layers.BatchNormalization(x)
        x = keras.layers.ReLU(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        return x

    def degradation_layer(self, x):
        residual = x
        for _ in range(4):
            x = keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='valid')(x)
            x = keras.layers.BatchNormalization(x)
            x = keras.layers.ReLU(x)
            x = keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='valid')(x)
            x = keras.layers.BatchNormalization(x)
            x = keras.layers.ReLU(x)
            x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = keras.layers.add(residual, x)
        return x

    def sep_convolution_layer(self, x):
        for _ in range(4):
            residual = x
            x = keras.layers.SeparableConvolution2D(filters=1024, kernel_size=(2, 2), strides=(2, 2),
                                                       padding='valid')(x)
            x = keras.layers.BatchNormalization(x)
            x = keras.layers.ReLU(x)
            x = keras.layers.SeparableConvolution2D(filters=1024, kernel_size=(2, 2), strides=(2, 2),
                                                       padding='valid')(x)
            x = keras.layers.BatchNormalization(x)
            x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
            x = keras.layers.add(residual, x)
        return x

    def heatmap_layer(self, x):
        for _ in range(2):
            x = keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='valid')(x)
            x = keras.layers.BatchNormalization(x)
            x = keras.layers.ReLU(x)
            x = keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='valid')(x)
            x = keras.layers.BatchNormalization(x)
            x = keras.layers.ReLU(x)
            x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        return x

    def multi_detect_layer(self, x):
        for _ in range(4):
            # bb convolution
            # softmax regression
            # convolution
            pass
        return x

    def forward(self, x):
        return self.multi_detect_layer(
            self.heatmap_layer(self.sep_convolution_layer(self.degradation_layer(self.init_layer(x)))))

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
