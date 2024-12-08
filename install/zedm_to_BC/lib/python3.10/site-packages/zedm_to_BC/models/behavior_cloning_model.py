from keras import layers, models
import tensorflow as tf

class ConvCfCModel(models.Model):
    def __init__(self, input_shape, output_size):
        super().__init__()
        self.input_layer = layers.Input(shape=input_shape)
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(output_size, activation='linear')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        outputs = self.fc(x)
        return outputs
