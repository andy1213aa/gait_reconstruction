import tensorflow as tf
from tensorflow.keras import layers, initializers

class View_Transform_Layer(tf.keras.Model):

    def __init__(self, view_number):
        super(View_Transform_Layer, self).__init__()
        self.f1 = layers.Dense(units = view_number)
    
    def call(self, inputs):
        gait_encode, view_onehot = inputs
        x = gait_encode + self.f1(view_onehot)
        return x

    def model(self, gait_encode_shape: int, view_onehot_shape: int) -> tf.keras.models:
        gait_encode_input = tf.keras.Input(shape=(gait_encode_shape), name = 'gait_encode_input')
        view_onehot_input = tf.keras.Input(shape=(view_onehot_shape), name= 'view_onehot_input')
        return tf.keras.models.Model(inputs=input, outputs = self.call([gait_encode_input, view_onehot_input]))