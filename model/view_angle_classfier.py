import tensorflow as tf
from tensorflow.keras import layers, initializers

class View_Angle_Classfier(tf.keras.Model):
    
    def __init__(self, view_dim):
        super(View_Angle_Classfier, self).__init__()
        self.f1 = layers.Dense(units = 128, name = "VAC_F1")
        self.leakyReLU1 = layers.LeakyReLU(name = "LeakyReLU1")
        self.f2 = layers.Dense(units = view_dim)
        self.softmax = tf.keras.layers.Softmax()    
    
    def call(self, input):
        x = self.f1(input)
        x = self.leakyReLU1(x)
        x = self.f2(x)
        x = self.softmax(x)
        return x

    def model(self, inputsize: int) -> tf.keras.models:
        input = tf.keras.Input(shape=inputsize, name='input_layer')
        
        return tf.keras.models.Model(inputs=input, outputs = self.call(input))

    