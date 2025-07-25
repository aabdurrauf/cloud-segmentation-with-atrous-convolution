# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:09:39 2024

@author: ammar

source: https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection/blob/master/Cloud-Net/augmentation.py
"""

import numpy as np
import skimage.transform as trans

"""
Some lines borrowed from: https://www.kaggle.com/sashakorekov/end-to-end-resnet50-with-tta-lb-0-93
"""


def rotate_clk_img_and_msk(img, msk):
    angle = np.random.choice(tuple(i for i in range(5, 50, 5)))
    img_o = trans.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
    msk_o = trans.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
    return img_o, msk_o

def rotate_cclk_img_and_msk(img, msk):
    angle = np.random.choice(tuple(i for i in range(-5, -50, -5)))
    img_o = trans.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
    msk_o = trans.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
    return img_o, msk_o

def flip_img_and_msk(img, msk):
    random_axis = np.random.choice((0, 1))
    img_o = np.flip(img, axis=random_axis)
    msk_o = np.flip(msk, axis=random_axis)
    return img_o, msk_o

def zoom_img_and_msk(img, msk):

    zoom_factor = np.random.choice((1.2, 1.5, 1.8, 2, 2.2, 2.5))  # currently doesn't have zoom out!
    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    img = trans.resize(img, (zh, zw), preserve_range=True, mode='symmetric')
    msk = trans.resize(msk, (zh, zw), preserve_range=True, mode='symmetric')
    region = np.random.choice((0, 1, 2, 3, 4))

    # zooming out
    if zoom_factor <= 1:
        outimg = img
        outmsk = msk

    # zooming in
    else:
        # bounding box of the clipped region within the input array
        if region == 0:
            outimg = img[0:h, 0:w]
            outmsk = msk[0:h, 0:w]
        if region == 1:
            outimg = img[0:h, zw - w:zw]
            outmsk = msk[0:h, zw - w:zw]
        if region == 2:
            outimg = img[zh - h:zh, 0:w]
            outmsk = msk[zh - h:zh, 0:w]
        if region == 3:
            outimg = img[zh - h:zh, zw - w:zw]
            outmsk = msk[zh - h:zh, zw - w:zw]
        if region == 4:
            marh = h // 2
            marw = w // 2
            outimg = img[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]
            outmsk = msk[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]

    # to make sure the output is in the same size of the input
    img_o = trans.resize(outimg, (h, w), preserve_range=True, mode='symmetric')
    msk_o = trans.resize(outmsk, (h, w), preserve_range=True, mode='symmetric')
    return img_o, msk_o