import tensorflow as tf
from tensorflow.keras import layers, initializers, Model

class Generator(Model):

    def __init__(self, k: float):
        super(Generator, self).__init__()

        self.conv1 = layers.Conv2DTranspose(filters = 256, 
            kernel_size = (3, 3), strides = 2, use_bias = False)    
        self.bn_1 = layers.BatchNormalization(name='BN1')
        self.leakyReLU1 = layers.LeakyReLU(name='leakyReLU1', alpha=0.01)

        self.conv2 = layers.Conv2DTranspose(filters = 128, 
            kernel_size = (3, 3), strides = 2, use_bias = False)    
        self.bn_2 = layers.BatchNormalization(name='BN2')
        self.leakyReLU2 = layers.LeakyReLU(name='leakyReLU2', alpha=0.01)

        self.conv3 = layers.Conv2DTranspose(filters = 64, 
            kernel_size = (3, 3), strides = 2, use_bias = False)    
        self.bn_3 = layers.BatchNormalization(name='BN3')
        self.leakyReLU3 = layers.LeakyReLU(name='leakyReLU3', alpha=0.01)

        self.conv4 = layers.Conv2DTranspose(filters = 32, 
            kernel_size = (3, 3), strides = 2, use_bias = False)    
        self.bn_4 = layers.BatchNormalization(name='BN4')
        self.leakyReLU4 = layers.LeakyReLU(name='leakyReLU4', alpha=0.01)

        self.conv5 = layers.Conv2DTranspose(filters = 1, 
            kernel_size = (3, 3), strides = 2, use_bias = False)    

        self.tanh = layers.Activation('tanh')
    
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
        x = self.conv5(x)
        x = self.tanh(x)
        return x


