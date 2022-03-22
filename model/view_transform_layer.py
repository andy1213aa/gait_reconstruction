import tensorflow as tf
from tensorflow.keras import layers, initializers

class view_transform_layer(layers):

    def __init__(self, k):
        super(view_transform_layer, self).__init__()
        self.