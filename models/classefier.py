import tensorflow as tf
import keras
from keras import layers
from keras import activations

class Attributor(keras.Sequential):
    def __init__(self, input_shape, hidden_size_list, num_classes):
        super(Attributor, self).__init__()
        self.add(layers.InputLayer(input_shape=input_shape))
        self.add(layers.Dense(hidden_size_list[0], activation= activations.Relu))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(hidden_size_list[1], activation= activations.Relu))
        self.add(layers.Dense(num_classes, activation= None))


