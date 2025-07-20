# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:10:37 2024

@author: ammar
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization,Activation, Dropout
from tensorflow import keras
# from tensorflow.keras.utils import plot_model
# import numpy as np

def fusion_block(operand):
    """
    feature fusion block :
        - add-multiply-add (AMA)
        - add
    """
    # version 1
    # a = keras.layers.add(operand)
    # m = keras.layers.multiply(operand)
    # x = keras.layers.add([a, m])
    
    # version 2
    x = keras.layers.add(operand)
    
    return x


def bn_activation(input_tensor, activation="relu"):
    """
    Batch_normalization layer before defined activation function
    """
    input_tensor = BatchNormalization(axis=3)(input_tensor)
    return Activation(activation)(input_tensor)


def contracting_block(input_tensor, filters, kernel_size, dilation, activation="relu"):
    """It adds a feedforward signal to the output of two following conv layers in contracting path
    """

    x1 = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x1 = bn_activation(x1, activation)

    x1 = Conv2D(filters, kernel_size, padding='same')(x1)
    x1 = bn_activation(x1, activation=activation)
    
    x1 = Conv2D(filters, kernel_size, padding='same')(x1)
    x1 = bn_activation(x1, activation=activation)

    x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(input_tensor)
    x2 = bn_activation(x2, activation)
    
    x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(x2)
    x2 = bn_activation(x2, activation)
    
    x2 = Conv2D(filters, (1, 1), padding='same')(x2)
    x2 = bn_activation(x2, activation=activation)
    
    # x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(x2)
    # x2 = bn_activation(x2, activation)

    filters_b = filters // 2
    kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernel size of (1,1) out of (3,3)

    x3 = Conv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
    x3 = bn_activation(x3, activation)

    x3 = concatenate([input_tensor, x3], axis=3)
    
    x4 = fusion_block([x1, x2, x3])
    
    return Activation(activation)(x4)


def bridge(input_tensor, filters, kernel_size, dilation, activation="relu"):
    """It is exactly like the identity_block plus a dropout layer. This block only uses in the valley of the UNet
    """

    x1 = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x1 = bn_activation(x1, activation)

    x1 = Conv2D(filters, kernel_size, padding='same')(x1)
    x1 = Dropout(.15)(x1)
    x1 = bn_activation(x1, activation=activation)

    x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(input_tensor)
    x2 = Dropout(.15)(x2)
    x2 = bn_activation(x2, activation)
    
    x2 = Conv2D(filters, (1, 1), padding='same')(x2)
    x2 = bn_activation(x2, activation=activation)
    
    # x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(x2)
    # x2 = bn_activation(x2, activation)

    filters_b = filters // 2
    kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernel size of (1,1) out of (3,3)

    x3 = Conv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
    x3 = bn_activation(x3, activation)
    
    x3 = Conv2D(filters_b, kernel_size_b, padding='same')(x3)
    x3 = bn_activation(x3, activation)

    x3 = concatenate([input_tensor, x3], axis=3)
    
    x4 = fusion_block([x1, x2, x3])
    
    return Activation(activation)(x4)


def conv_block_exp_path(input_tensor, filters, kernel_size, dilation, activation="relu"):
    """
    Convolution part inside each expanding path's block
    """

    x1 = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x1 = bn_activation(x1, activation)

    x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(x1)
    x2 = bn_activation(x2, activation)
    
    x3 = Conv2D(filters, (1, 1), padding='same')(x2)
    x3 = bn_activation(x3, activation)

    return x3


# def feed_fodward_block(input_tensors, pure_ff, activation="relu"):
#     """It improves the skip connection by using previous layers feature maps
#     """
    
#     x_list = []
#     t = 1
#     for i in range(len(input_tensors)):
#         for ix in range(t):
#             if ix == 0:
#                 x = input_tensors[i]
#             x = concatenate([x, input_tensors[i]], axis=3)
        
#         ps = 2**(i+1)
#         x = MaxPooling2D(pool_size=(ps, ps))(x)
        
#         x_list.append(x)
#         t = t * 2 + 1
    
#     if len(input_tensors) > 1:
#         for ix in range(len(x_list)-1):
#             if ix == 0:
#                 ama = fusion_block([x_list[0], x_list[1]])
#             else:
#                 ama = fusion_block([ama, x_list[ix+1]])
#     else:
#         ama = x_list[0]
    
#     # print('shpae ama:', np.shape(ama))
#     # print('shape x0:', np.shape(x0))
#     # print('shape pure_ff:', np.shape(pure_ff), '\n')    
    
#     ama = fusion_block([ama, pure_ff])
    
#     return Activation(activation)(ama)


def extracting_block(prev_tensor, skip_layer, filters, kernel_size=(2, 2), strides=(2, 2), dilation=(2, 2), activation="relu"):
    
    convT = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')(prev_tensor)
    up = concatenate([convT, skip_layer], axis=3)

    conv = conv_block_exp_path(up, filters, kernel_size, dilation, activation)
    out = keras.layers.add([conv, convT])
    
    return out
    
    
def build_model(input_rows=192, input_cols=192, num_of_channels=4, num_of_classes=1, activation="relu"):
    
    inputs = Input((input_rows, input_cols, num_of_channels))
    
    
    # contracting part
    conv0 = Conv2D(16, (3, 3), activation=activation, padding='same')(inputs)
    
    conv1 = contracting_block(conv0, 32, (3, 3), (2, 2), activation)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = contracting_block(pool1, 64, (3, 3), (2, 2), activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = contracting_block(pool2, 128, (3, 3), (2, 2), activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = contracting_block(pool3, 256, (3, 3), (2, 2), activation)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = contracting_block(pool4, 512, (3, 3), (2, 2), activation)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv6 = bridge(pool5, 1024, (3, 3), (2, 2), activation)
    
    
    # expanding part
    extract7 = extracting_block(conv6, conv5, 512)
    
    extract8 = extracting_block(extract7, conv4, 256)
    
    extract9 = extracting_block(extract8, conv3, 128)
    
    extract10 = extracting_block(extract9, conv2, 64)
    
    extract11 = extracting_block(extract10, conv1, 32)
    
    outputs = Conv2D(num_of_classes, (1, 1), activation='sigmoid')(extract11)
    
    return Model(inputs=[inputs], outputs=[outputs])
    
    
model_name = "my_model_08"
# model = build_model()
# model.summary()



