from tensorflow import keras

class Net(keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=136, kernel_size=2, strides=2)
        self.pool = keras.layers.GlobalAveragePooling2D()
        # self.flat = keras.layers.Flatten()
        # self.fcl1 = keras.layers.Dense(10, activation = 'relu')
        # self.fcl2 = keras.layers.Dense(136, activation = 'softmax')

    def call(self, inputs):
        return self.pool(self.conv1(inputs))
        #return self.fcl2(self.fcl1(self.flat(self.conv1(inputs))))