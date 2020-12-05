import math
import sys

import cv2
import copy
import numpy as np
import random
from keras.engine import Layer
from keras import backend as K, Input
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense, Dropout


class Config:

    def __init__(self):
        # Base pretrained network that we are using
        self.network = "vgg"
        self.model_path = None  # path to vgg weights

        # original paper has 128, 256, 512 but we scale down for training time
        self.anchor_box_scales = [64, 128, 256]
        # ratios are 1:1, 1:2, 2:1 from the paper
        self.anchor_box_ratios = [[1, 1], [1.0 / math.sqrt(2), 2.0 / math.sqrt(2)],
                                  [2.0 / math.sqrt(2), 1.0 / math.sqrt(2)]]

        # resizing the smallest side of the image
        # the paper has 600, but we scale further for training time
        self.img_size = 300

        # # of ROIs at once
        self.num_rois = 4
        self.rpn_stride = 16  # can tune

        self.rpn_min_overlap = .3
        self.rpn_max_overlap = .7


# Calculating Intersection over Union (IoU) metric
def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

