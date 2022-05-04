import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa

class Discriminator(tf.keras.Model):

    # def __init__(self, num_class: int, num_pei_channel: int, view_dim: int):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 =  tfa.layers.SpectralNormalization(layers.Conv2D(filters=32,
                                   kernel_size=(7, 7), strides=2, padding='same',
                                   name='conv1',  use_bias=False))
        self.leakyReLU1 = layers.LeakyReLU(name='leakyReLU1')
        self.conv2 =  tfa.layers.SpectralNormalization(layers.Conv2D(filters=64,
                                   kernel_size=(5, 5), strides=2, padding='same',
                                   name='conv2',  use_bias=False))
        self.leakyReLU2 = layers.LeakyReLU(name='leakyReLU2')
        self.conv3 =  tfa.layers.SpectralNormalization(layers.Conv2D(filters=128,
                                   kernel_size=(5, 5), strides=2, padding='same',
                                   name='conv3',  use_bias=False))
        self.leakyReLU3 = layers.LeakyReLU(name='leakyReLU3')
        self.conv4 =  tfa.layers.SpectralNormalization(layers.Conv2D(filters=256,
                                   kernel_size=(3, 3), strides=2, padding='same',
                                   name='conv4',  use_bias=False))
        self.flatten =  layers.Flatten()
        self.leakyReLU4 = layers.LeakyReLU(name='leakyReLU4')
        # self.f1 = layers.Dense(
        #     units=num_class+num_pei_channel+view_dim, name='F1')
        self.f1 =  tfa.layers.SpectralNormalization(layers.Dense(units=1, name='F1'))

    def call(self, input):

        x = self.conv1(input)
        x = self.leakyReLU1(x)
        x = self.conv2(x)
        x = self.leakyReLU2(x)
        x = self.conv3(x)
        x = self.leakyReLU3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.leakyReLU4(x)
        x = self.f1(x)
        return x

    def model(self, inputsize: int) -> tf.keras.models:
        input = tf.keras.Input(
            shape=(inputsize[0], inputsize[1], 1), name='input_layer')

        return tf.keras.models.Model(inputs=input, outputs=self.call(input))
