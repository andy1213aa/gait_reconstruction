import tensorflow as tf
from tensorflow.keras import layers, initializers
class Encoder(layers.Layer):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = layers.Conv2D(32, 
            kernel_size = (7, 7), strides=2, padding='same', 
            name = 'conv1',  use_bias=False)

        self.bn_1 = layers.BatchNormalization(name='BN1')
        self.leakyReLU1 = layers.LeakyReLU(name='leakyReLU1')

        self.conv2 = layers.Conv2D(64, 
            kernel_size = (5, 5), strides=2, 
            padding='same', name = 'conv2',  use_bias=False)

        self.bn_2 = layers.BatchNormalization(name='BN2')
        self.leakyReLU2 = layers.LeakyReLU(name='leakyReLU2')

        self.conv3 = layers.Conv2D(128, 
            kernel_size = (5, 5), strides=2, 
            padding='same', name = 'conv3',  use_bias=False)

        self.bn_3 = layers.BatchNormalization(name='BN3')
        self.leakyReLU3 = layers.LeakyReLU(name='leakyReLU3')

        self.conv4 = layers.Conv2D(256, 
            kernel_size = (3, 3), strides=2, 
            padding='same', name = 'conv4',  use_bias=False)

        self.bn_4 = layers.BatchNormalization(name='BN4')
        self.leakyReLU4 = layers.LeakyReLU(name='leakyReLU4')

        self.mean_pooling = layers.AveragePooling2D(name = 'mean_pooling')

        self.f1 = layers.Dense(4096*128, name = 'F1')


        
        

    def call(self, input):

        x = self.conv1(input)
        x = self.bn_1(x)
        x = self.leakyReLU1(x)

        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.leakyReLU2(x)

        x = self.conv3(x)
        x = self.bn_3(x)
        x = self.leakyReLU3(x)

        x = self.conv4(x)
        x = self.bn_4(x)
        x = self.leakyReLU4(x)

        x = self.mean_pooling(x)
        x = self.f1(x)
        return x


            
