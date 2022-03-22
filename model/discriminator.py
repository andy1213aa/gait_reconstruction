import tensorflow as tf
from tensorflow.keras import layers, initializers


class Discriminator(tf.keras.Model):

    def __init__(self, k: float):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(32,
                                   kernel_size=(7, 7), strides=2, padding='same',
                                   name='conv1',  use_bias=False)
        self.leakyReLU1 = layers.LeakyReLU(name='leakyReLU1')
        self.conv2 = layers.Conv2D(64,
                                   kernel_size=(5, 5), strides=2, padding='same',
                                   name='conv2',  use_bias=False)
        self.leakyReLU2 = layers.LeakyReLU(name='leakyReLU2')
        self.conv3 = layers.Conv2D(128,
                                   kernel_size=(5, 5), strides=2, padding='same',
                                   name='conv3',  use_bias=False)
        self.leakyReLU3 = layers.LeakyReLU(name='leakyReLU3')
        self.conv4 = layers.Conv2D(256,
                                   kernel_size=(3, 3), strides=2, padding='same',
                                   name='conv4',  use_bias=False)
        self.leakyReLU4 = layers.LeakyReLU(name='leakyReLU4')
        self.f1 = layers.Dense(4096*k, name='F1')

    def call(self, input):

        x = self.conv1(input)
        x = self.leakyReLU1(x)
        x = self.conv2(x)
        x = self.leakyReLU2(x)
        x = self.conv3(x)
        x = self.leakyReLU3(x)
        x = self.conv4(x)
        x = self.leakyReLU4(x)
        x = self.f1(x)
        return x
    

    def model(self, inputsize: int) -> tf.keras.models:
        volumeInput = tf.keras.Input(shape=(inputsize, inputsize, inputsize, 1), name='volume')
        parameterInput = tf.keras.Input(shape=(1, 3), name='parameter')

        return tf.keras.models.Model(inputs=[parameterInput, volumeInput], outputs = self.call([parameterInput, volumeInput]))
