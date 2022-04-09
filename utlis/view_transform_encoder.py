import numpy as np

def view_angle_onehot(angle1: int, angle2: int, view_dim: int) -> np.array:
    
    '''
    We will use a NN called view_angle_classfier to classfy what angle is it in testing phase.
    However, we could not use this NN while training, so we build a encoder when training which transform the angle into one hot encoding.

    KEYARG:

    angle1: input image angle
    angle2: desired output angle

    possible angle: [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]

    '''

    # Use simple way to do one hot for now.
    # Need to be fixed in the future.
    # 2022/04/08
    angle1_onehot = angle1 // 18
    angle2_onehot = angle2 // 18

    result = np.zeros(11, 1) #11 posible angle
    result[angle1_onehot] = 1
    result[angle2_onehot] = 1


    return result

