import tensorflow as tf
import numpy as np
import datetime
import os
from shutil import copytree, copyfile
from pathlib import Path

class Save_Model(tf.keras.callbacks.Callback):
    def __init__(self, encoder, view_transform_layer, gen, dis, mode='min', save_weights_only=False):
        super(Save_Model, self).__init__()
        self.encoder = encoder
        self.view_transform_layer = view_transform_layer
        self.gen = gen
        self.dis = dis

        # setting directory of saving weight
        # self.dataSetConfig = dataSetConfig

        # biggest better or lowest better
        self.mode = mode
        # save type
        self.save_weights_only = save_weights_only
        if mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf

        self.counter = 0
        self.training = True
        self.epoch = 1

        startingTime = datetime.datetime.now()
        startingDate = f'{startingTime.year}_{startingTime.month}_{startingTime.day}' + \
            '_'+f'{startingTime.hour}_{startingTime.minute}'

        os.mkdir(f"./log/{startingDate}")

        self.encoderDir = f"./log/{startingDate}/encoder/"
        self.view_transform_layerDir = f"./log/{startingDate}/view_transform_layer/"
        self.genDir = f"./log/{startingDate}/gen/"
        self.disDir = f"./log/{startingDate}/dis/"

        if not os.path.isdir(f"./log/{startingDate}/encoder/"):
            os.mkdir(f"./log/{startingDate}/encoder/")

        if not os.path.isdir(f"./log/{startingDate}/view_transform_layer/"):
            os.mkdir(f"./log/{startingDate}/view_transform_layer/")

        if not os.path.isdir(f"./log/{startingDate}/gen/"):
            os.mkdir(f"./log/{startingDate}/gen/")

        if not os.path.isdir(f"./log/{startingDate}/dis/"):
            os.mkdir(f"./log/{startingDate}/dis/")



        work_dir = os.path.abspath('')
        copytree(f'{work_dir}/model',
                 f'./log/{startingDate}/model')
        copytree(f'{work_dir}/utlis',
                 f'./log/{startingDate}/utlis')

        copyfile(
            f'{work_dir}/main.py', f'./log/{startingDate}/main.py')

    def save(self):
        if self.save_weights_only:
            self.gen.save_weights(self.genDir + "trained_ckpt")
            self.dis.save_weights(self.disDir + "trained_ckpt")
        else:
            self.encoder.save(self.encoderDir + "trained_ckpt")
            self.view_transform_layer.save(
                self.view_transform_layerDir + "trained_ckpt")
            self.gen.save(self.genDir + "trained_ckpt")
            self.dis.save(self.disDir + "trained_ckpt")

    # def save_config(self, monitor_value):
    #     saveLogTxt = f"""
    # Parameter Setting
    # =======================================================
    # DataSet: { self.dataSetConfig['dataSet']}
    # DataShape: ({ self.dataSetConfig['length']}, { self.dataSetConfig['width']}, {self.dataSetConfig['height']})
    # DataSize: {self.dataSetConfig['datasize']}
    # TrainingSize: { self.dataSetConfig['trainSize']}
    # TestingSize: { self.dataSetConfig['testSize']}
    # BatchSize: { self.dataSetConfig['batchSize']}
    # =======================================================
    # Training log
    # =======================================================
    # Training start: { self.dataSetConfig['startingTime']}
    # Training stop: {datetime.datetime.now()}
    # Training epoch: {self.epoch}
    # Root Mean Square Error: {monitor_value}%
    # =======================================================
    # """
    #     with open(self.dataSetConfig['logDir']+'config.txt', 'w') as f:
    #         f.write(saveLogTxt)

    def on_epoch_end(self, monitor_value=0, logs=None):
        # read monitor value from logs
        # monitor_value = logs.get(self.monitor)
        # Create the saving rule

        # if self.mode == 'min' and monitor_value < self.best:

        #     self.best = monitor_value
        #     self.counter = 0
        # elif self.mode == 'max' and monitor_value > self.best:

        #     self.best = monitor_value
        #     self.counter = 0
        # else:
        #     self.counter += 1
        #     if self.counter >= self.dataSetConfig['stopConsecutiveEpoch']:
        #         self.save_model()
        #         self.save_config(monitor_value)
        #         self.training = False
        self.epoch += 1
