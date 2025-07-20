# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:44:18 2024

@author: ammar
"""
# AtrousCloud-Net

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization,Activation, Dropout
from tensorflow import keras


def bn_activation(input_tensor, activation="relu"):
    """
    Batch_normalization layer before activation function
    """
    input_tensor = BatchNormalization(axis=3)(input_tensor)
    return Activation(activation)(input_tensor)


def contracting_block(input_tensor, filters, kernel_size, dilation, activation="relu"):
    """
    feedforward signal to the output of two following conv layers 
    and an atrous conv layer in contracting path
    """

    x1 = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x1 = bn_activation(x1, activation)

    x1 = Conv2D(filters, kernel_size, padding='same')(x1)
    x1 = bn_activation(x1, activation)

    x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(input_tensor)
    x2 = bn_activation(x2, activation)
    
    x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(x2)
    x2 = bn_activation(x2, activation)

    filters_b = filters // 2
    kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)

    x3 = Conv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
    x3 = bn_activation(x3, activation)

    x3 = concatenate([input_tensor, x3], axis=3)
    
    x4 = keras.layers.add([x1, x2, x3])
    
    return Activation(activation)(x4)


def bridge(input_tensor, filters, kernel_size, dilation, activation="relu"):
    """
    exactly like the same as the contracting_block 
    plus a dropout layer. This block only uses in the valley of the UNet
    """

    x1 = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x1 = bn_activation(x1, activation)

    x1 = Conv2D(filters, kernel_size, padding='same')(x1)
    x1 = Dropout(.15)(x1)
    x1 = bn_activation(x1, activation)

    x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(input_tensor)
    x2 = bn_activation(x2, activation)
    
    x2 = Conv2D(filters, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(x2)
    x2 = Dropout(.15)(x2)
    x2 = bn_activation(x2, activation)
    
    filters_b = filters // 2
    kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2) 

    x3 = Conv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
    x3 = bn_activation(x3, activation)

    x3 = concatenate([input_tensor, x3], axis=3)
    
    x4 = keras.layers.add([x1, x2, x3])
    
    return Activation(activation)(x4)


def conv_block_exp_path(input_tensor, filters, kernel_size, dilation, activation="relu"):
    """
    Convolution part inside each expanding path's block
    using Conv layer and Atrous Conv layer
    """

    x1 = Conv2D(filters//2, kernel_size, padding='same')(input_tensor)
    x1 = bn_activation(x1, activation)

    
    x2 = Conv2D(filters//2, kernel_size, strides=(1, 1), dilation_rate=dilation, padding='same')(input_tensor)
    x2 = bn_activation(x2, activation)
    
    x3 = concatenate([x1, x2], axis=3)

    return x3


def expanding_block(prev_tensor, skip_layer, filters, kernel_size_T=(2, 2), 
                     kernel_size=(3, 3), strides=(2, 2), dilation=(2, 2), activation="relu"):
    
    convT = Conv2DTranspose(filters, kernel_size_T, strides=strides, padding='same')(prev_tensor)
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
    extract7 = expanding_block(conv6, conv5, 512)
    
    extract8 = expanding_block(extract7, conv4, 256)
    
    extract9 = expanding_block(extract8, conv3, 128)
    
    extract10 = expanding_block(extract9, conv2, 64)
    
    extract11 = expanding_block(extract10, conv1, 32)
    
    outputs = Conv2D(num_of_classes, (1, 1), activation='sigmoid')(extract11)
    
    return Model(inputs=[inputs], outputs=[outputs])
    
    
model_name = "AtrousCloud-Net"
# model = build_model()
# model.summary()



