from tensorflow import keras

class HRNetLearningRate(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        if step>=30:
            return self.initial_learning_rate/10
        elif step>=50:
            return self.initial_learning_rate/100
        return self.initial_learning_rate