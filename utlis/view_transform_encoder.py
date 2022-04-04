import numpy as np

def view_transform_encoder(angle1: int, angle2: int, view_dim: int) -> np.array:
    '''
    We will use a NN called view_angle_classfier to classfy what angle is it in testing phase.
    However, we could not use this NN while training, so we build a encoder when training which transform the angle into one hot encoding.


    [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]

    
    '''

    one_hot = np.zeros((view_dim))

