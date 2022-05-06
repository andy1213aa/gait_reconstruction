import numpy as np
import tensorflow as tf

def random_view_angle(angles: np.array) -> np.array:
    
    '''
    We will use a NN called view_angle_classfier to classfy what angle is it in testing phase.
    However, we could not use this NN while training, so we build a encoder when training which transform the angle into one hot encoding.

    KEYARG:

    angles: possible angle. Ex. [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]

    '''

    # Use simple way to do one hot for now.
    # Need to be fixed in the future.
    # 2022/04/08
    
    
    sample_num = 2

    idxs = tf.range(tf.shape(angles)[0])
    ridxs = tf.random.shuffle(idxs)[:sample_num]
    
    

    return ridxs[0], ridxs[1]

