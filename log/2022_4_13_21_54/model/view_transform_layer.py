import tensorflow as tf
from tensorflow.keras import layers, initializers

class View_Transform_Layer(tf.keras.Model):

    def __init__(self, view_number):
        super(View_Transform_Layer, self).__init__()
        self.f1 = layers.Dense(units = view_number)
        self.identity = layers.Layer()
        self.sum = tf.math.reduce_sum

    def call(self, inputs):
        gait_encode, view_onehot = inputs
        x = self.identity(gait_encode)
        y = self.f1(view_onehot)
        out = self.sum(x+y)

        return out

    def model(self, gait_encode_shape: int, view_onehot_shape: int) -> tf.keras.models:
        gait_encode_input = tf.keras.Input(shape=(gait_encode_shape), name = 'gait_encode_input')
        view_onehot_input = tf.keras.Input(shape=(view_onehot_shape), name= 'view_onehot_input')
        return tf.keras.models.Model(inputs=[gait_encode_input, view_onehot_input], outputs = self.call([gait_encode_input, view_onehot_input]))