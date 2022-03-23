import numpy as np


def Period_Detection(gallery: list, gallery_shape: tuple, alpha=0.470, beta=0.961) -> np.array:
    '''
    hip, knee, and ankle are 0.470, 0.715, 0.961
    '''
    start = round(gallery_shape[0] * alpha)
    end = round(gallery_shape[0] * beta)

    period_result = np.empty(len(gallery))

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
