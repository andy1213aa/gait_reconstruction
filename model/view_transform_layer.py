import tensorflow as tf
from tensorflow.keras import layers, initializers

class View_Transform_Layer(tf.keras.Model):

    def __init__(self, view_number):
        super(View_Transform_Layer, self).__init__()
        self.f1 = layers.Dense(units = view_number)
    
    def call(self, gait_encode, view_onehot):
        x = gait_encode + self.f1(view_onehot)
        return x

    def model(self, inputsize: int, ) -> tf.keras.models:
        input = tf.keras.Input(shape=(inputsize), name = 'input_layer')
        return tf.keras.models.Model(inputs=input, outputs = self.call(input))