import numpy as np
from absl import flags
import os
import matplotlib.pyplot as plt
import cv2
FLAGS = flags.FLAGS

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('dataset', None, 'dataset name.')
# flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
# flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
# flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')


def main():

    if not FLAGS.dataset:
        print("Please give correct dataset name. ")


def Period_Detection(gallery: np.array, gallery_shape: tuple,
                     alpha=0.470, beta=0.961) -> np.array:
    '''
    From the early research, hip, knee, and ankle are 0.470, 0.715, 0.961 ratio from a human height.
    In this function, we caculate the mean of the ppl' moving(i.e. the region from hip to ankle. )
    
    Input arg:
        gallery: np.array. A frame collection that taken from human's gait. 
        gallery_shape: tuple. The shape of each gallery image.
        alpha, beta: float. The ratio that you want to slice from a single ppl.

    Output: 

        period_result: nparray. The size of output nparray will be the shape[0] of gallery, 
        which each element represents the mean of ppl moving in a frame. 

    Note:
        We assume the input gallery have been preprocessing into grayscale image, whcih 
        have also seperate the foreground and background. 

    '''
    start = round(gallery_shape[0] * alpha)
    end = round(gallery_shape[0] * beta)

    period_result = np.empty(gallery.shape[0])

    for i, img in enumerate(gallery):
        move_region_aggregate = 0

        for h in range(start, end):

            nonzero_index = list(np.nonzero(img[h])[0])

            if not nonzero_index:
                continue
            left_most, right_most = nonzero_index[0], nonzero_index[-1]

            move_region_aggregate = move_region_aggregate + \
                (right_most - left_most)

        move_region_aggregate /= float((end - start + 1))

        period_result[i] = move_region_aggregate
    if period_result.shape[0] == 1:
        return period_result
    period_result = (period_result - np.min(period_result)) / \
        np.ptp(period_result)

    return period_result


def GEI(images: np.array, resize: tuple) -> list:

    def mass_center(img, is_round=True):
        Y = img.mean(axis=1)
        X = img.mean(axis=0)
        Y_ = np.sum(np.arange(Y.shape[0]) * Y)/np.sum(Y)
        X_ = np.sum(np.arange(X.shape[0]) * X)/np.sum(X)
        if is_round:
            return int(round(X_)), int(round(Y_))
        return X_, Y_

    def image_extract(img, newsize):
        x_s = np.where(img.mean(axis=0) != 0)[0].min()
        x_e = np.where(img.mean(axis=0) != 0)[0].max()

        y_s = np.where(img.mean(axis=1) != 0)[0].min()
        y_e = np.where(img.mean(axis=1) != 0)[0].max()

        x_c, _ = mass_center(img)
    #     x_c = (x_s+x_e)//2
        x_s = x_c-newsize[1]//2
        x_e = x_c+newsize[1]//2
        img = img[y_s:y_e, x_s if x_s > 0 else 0:x_e if x_e <
                  img.shape[1] else img.shape[1]]
        return cv2.resize(img, newsize)

    images = [image_extract(i, resize) for i in images]
    gei = np.mean(images, axis=0)
    plt.imshow(gei)
    plt.show()


def Tk(m=1, nc=5) -> list:
    result = []
    for k in range(1, nc+1):
        tmp = [k/(nc+1) - (m/2), k/(nc+1) + m/2]
        result.append(tmp)
    return result


if __name__ == '__main__':
    main()
