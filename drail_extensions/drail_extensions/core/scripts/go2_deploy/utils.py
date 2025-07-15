import numpy as np


def remap(val, min1, max1, min2, max2):
    span1 = max1 - min1
    span2 = max2 - min2
    scaled = (val - min1) / span1
    return np.clip(min2 + (scaled * span2), min2, max2)
