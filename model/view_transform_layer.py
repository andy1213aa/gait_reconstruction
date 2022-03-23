import tensorflow as tf
from tensorflow.keras import layers, initializers

class View_Transform_Layer(tf.keras.Model):

    def __init__(self, k):
        super(View_Transform_Layer, self).__init__()
        self.f1 = layers.Dense(units = k)
    
    def call(self, input, view_encode):
        x = input + self.f1(view_encode)
        return x

    def model(self, inputsize: int, ) -> tf.keras.models:
        input = tf.keras.Input(shape=(inputsize), name = 'input_layer')
        return tf.keras.models.Model(inputs=input, outputs = self.call(input))