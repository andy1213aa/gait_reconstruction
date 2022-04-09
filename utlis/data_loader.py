from re import M
from xmlrpc.client import boolean
import numpy as np
import tensorflow as tf


class GenericTFLoader():
    '''
    load tfrecord data.
    '''

    def __init__(self, config):
        self._config = config

    def read(self):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError

    @classmethod
    def generate_loader(cls, config):
        raise NotImplementedError


class CasiaB(GenericTFLoader):
    super().__init__()

    def read(self, batch_size: int, shuffle: boolean) -> tf.Dataset:

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_set = tf.data.TFRecordDataset(self._config['dataSetDir'])
        data_set = data_set.map(_parse_function, num_parallel_calls=AUTOTUNE)
        if shuffle:
            data_set = data_set.shuffle(
                self._config['trainSize'], reshuffle_each_iteration=True)
        data_batch = data_set.batch(batch_size, drop_remainder=True)
        data_batch = data_batch.prefetch(buffer_size=AUTOTUNE)

        return data_batch

    def parse(self, example_proto: tf.Dataset) -> tf.Dataset:

        

        features = tf.io.parse_single_example(
            example_proto,
            features={feature: tf.io.FixedLenFeature([], dtype) for feature, dtype in self._config('features')
            }
        )
        img = features['image']

        img = tf.io.decode_raw(data, tf.float32)
        img = tf.reshape(data, )
        
        data = (data - mini) / (maxi-mini)
        data = data*2 - 1 #rescale the value range to [-1, 1].
     
        
        P1 = tf.reshape(P1, [1])
        P2 = tf.reshape(P2, [1])
        P3 = tf.reshape(P3, [1])
        P = tf.stack([P1, P2, P3], axis=1)
        return P, data

    @classmethod
    def generate_loader(cls, config):
        yield cls(config)
