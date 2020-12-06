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


# Creating a Region of Interest (ROI) pooling layer for 2D inputs
# Spatial Pyramid Pooling in Deep Conv. Nets paper
class RoiPooling2D(Layer):

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_dim_ordering()  # "tf", using TensorFlow backend
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        img = x[0]  # image with (rows, cols, channels) as shape
        rois = x[1]  # roi with (num_rois, 4) as shape, ordering is (x, y, w, h)

        input_shape = K.shape(img)
        outputs = []

        for idx in range(self.num_rois):
            x = K.cast(rois[0, idx, 0], 'int32')
            y = K.cast(rois[0, idx, 1], 'int32')
            w = K.cast(rois[0, idx, 2], 'int32')
            h = K.cast(rois[0, idx, 3], 'int32')

            # resize ROI to pooling size
            resized = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(resized)

        # reshape to (1, num_rois, pool size, pool size, # of channels)
        final_out = K.concatenate(outputs, axis=0)
        final_out = K.reshape(final_out, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        final_out = K.permute_dimensions(final_out, (0, 1, 2, 3, 4))

        return final_out

    def get_config(self):
        config = {"pool_size": self.pool_size,
                  "num_rois": self.num_rois}
        base_config = super(RoiPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Creating the vgg16 model
# based on vgg16 model architecture
def vgg_base(input_tensor=None, trainable=False):
    input_shape = (None, None, 3)

    # validate the input tensor and make input layer
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # 1 Block
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(img_input)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # 2 Block
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # 3 Block
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # 4 Block
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # 5 Block
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)

    return x

