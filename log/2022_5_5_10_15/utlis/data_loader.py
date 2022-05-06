from re import M
from xmlrpc.client import boolean
import numpy as np
import tensorflow as tf
from utlis.view_transform_encoder import random_view_angle


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

    # @classmethod
    # def generate_loader(cls, loader_subclasses, config):
    #     loader_collection = []
    #     for loader in loader_subclasses:
    #         loader_subclass()


# class CasiaB(GenericTFLoader):

#     def __init__(self, config):
#         self._config = config

#     def read(self, batch_size: int, shuffle: boolean) -> tf.Dataset:

#         AUTOTUNE = tf.data.experimental.AUTOTUNE
#         data_set = tf.data.TFRecordDataset(self._config['dataSetDir'])
#         data_set = data_set.map(self.parse, num_parallel_calls=AUTOTUNE)
#         if shuffle:
#             data_set = data_set.shuffle(
#                 self._config['trainSize'], reshuffle_each_iteration=True)
#         data_batch = data_set.batch(batch_size, drop_remainder=True)
#         data_batch = data_batch.prefetch(buffer_size=AUTOTUNE)

#         return data_batch

#     def parse(self, example_proto: tf.Dataset) -> tf.Dataset:

#         pass

#     @classmethod
#     def generate_loader(cls, config):
#         yield cls(config)


class OU_MVLP(GenericTFLoader):

    def __init__(self, config):
        self._config = config

    def read(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_set = tf.data.TFRecordDataset(
            self._config['training_info']['tfrecord_path'])
        data_set = data_set.map(self.parse, num_parallel_calls=AUTOTUNE)

        data_set = data_set.shuffle(
            64, reshuffle_each_iteration=self._config['training_info']['shuffle'])
        data_batch = data_set.batch(
            self._config['training_info']['batch_size'], drop_remainder=True)
        data_batch = data_batch.prefetch(buffer_size=AUTOTUNE)
        return data_batch

    def parse(self, example_proto):

        features = tf.io.parse_single_example(
            example_proto,
            features={key: tf.io.FixedLenFeature(
                [], self._config['feature'][key]) for key in self._config['feature']}

        )
        images = features['images']
        angles = features['angles']
        angles = tf.io.decode_raw(angles, tf.int32)

        # random select 2 angle
        # Here we return is the idx of angle
        # Note that not every data have all angle, so we used shaped to random choice
        random_angle = tf.random.shuffle(range(tf.shape(angles)[0]))[:2]

        # We still need to use maximun value of angle numbers to build onehot
        onehot_list = tf.one_hot(tf.range(
            self._config['resolution']['angle_nums']), self._config['resolution']['angle_nums'])

        angle_onehot = onehot_list[random_angle[0]] + \
            onehot_list[random_angle[1]]

        images = tf.io.decode_raw(images, tf.float32)
        images = (images-127.5)/127.5
        images = tf.reshape(images, [tf.shape(angles)[0], 128, 88, 3])

        random_select_image_1 = tf.image.resize(tf.image.rgb_to_grayscale(images[random_angle[0]]), [
                                                self._config['resolution']['height'], self._config['resolution']['width']])
        random_select_image_2 = tf.image.resize(tf.image.rgb_to_grayscale(images[random_angle[1]]), [
                                                self._config['resolution']['height'], self._config['resolution']['width']])

        return [features['subject'],
                angle_onehot, random_select_image_1, random_select_image_2]
