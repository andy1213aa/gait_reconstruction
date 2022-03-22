import tensorflow as tf
from tensorflow.keras import layers, initializers
class View_Angle_Classfier(layers.Layer):
    
    def __init__(self, k):
        super(View_Angle_Classfier, self).__init__()
        self.f1 = layers.Dense(128*128, name = "VAC_F1")
        self.leakyReLU1 = layers.LeakyReLU(name = "LeakyReLU1")
        self.f2 = layers.Dense(128*k)
        self.softmax = tf.keras.layers.Softmax()    
    

    def call(self, input):
        x = self.f1(input)
        x = self.leakyReLU1(x)
        x = self.f2(x)
        x = self.softmax(x)
        return x

    