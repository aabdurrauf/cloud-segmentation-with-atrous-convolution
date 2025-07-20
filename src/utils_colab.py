# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 00:50:36 2024

@author: ammar
"""

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from tqdm import tqdm
import os

def get_input_image(name_list, is_train=True):
    images = []
    masks = []
    test_ids = []
    
    for file_name in tqdm(name_list['name'], miniters=1000):
        red = 'red_' + file_name
        blue = 'blue_' + file_name
        green = 'green_' + file_name
        nir = 'nir_' + file_name
        
        if is_train:
            type_dir = 'train'
            image_files = []
            mask = 'gt_' + file_name
            mask_file = 'train_gt/train_gt/' + '{}.TIF'.format(mask)
            masks.append(mask_file)
            
        else:
            
            type_dir = 'test'
            image_files = []
            file_id = '{}.TIF'.format(file_name)
            test_ids.append(file_id)
            
        red_image = os.path.join(type_dir + '_red', type_dir + '_red', '{}.TIF'.format(red))
        green_image = os.path.join(type_dir + '_green', type_dir + '_green', '{}.TIF'.format(green))
        blue_image = os.path.join(type_dir + '_blue', type_dir + '_blue', '{}.TIF'.format(blue))
        nir_image = os.path.join(type_dir + '_nir', type_dir + '_nir', '{}.TIF'.format(nir))
        
        image_files.append(red_image)
        image_files.append(green_image)
        image_files.append(blue_image)
        image_files.append(nir_image)
        
        images.append(image_files)
        
    if is_train:
        return images, masks
    else:
        return images, test_ids
        


smooth = 0.0000001
def jacc_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))        


class ADAMLearningRateTracker(Callback):
    """It prints out the last used learning rate after each epoch (useful for resuming a training)
    original code: https://github.com/keras-team/keras/issues/7874#issuecomment-329347949
    """

    def __init__(self, end_lr):
        super(ADAMLearningRateTracker, self).__init__()
        self.end_lr = end_lr

    def on_epoch_end(self, epoch, logs={}):  # works only when decay in optimizer is zero
        optimizer = self.model.optimizer
        # t = K.cast(optimizer.iterations, K.floatx()) + 1
        # lr_t = K.eval(optimizer.lr * (K.sqrt(1. - K.pow(optimizer.beta_2, t)) /
        #                               (1. - K.pow(optimizer.beta_1, t))))
        # print('\n***The last Actual Learning rate in this epoch is:', lr_t,'***\n')
        print('\n***The last Basic Learning rate in this epoch is:', K.eval(optimizer.lr), '***\n')
        # stops the training if the basic lr is less than or equal to end_learning_rate
        if K.eval(optimizer.lr) <= self.end_lr:
            print("training is finished")
            self.model.stop_training = True
        
        
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

def show_images(img_files, mask_file, height=192, width=192, max_possible_input_value=65535):
    red = imread(img_files[0])
    green = imread(img_files[1])
    blue = imread(img_files[2])
    nir = imread(img_files[3])
    
    mask = imread(mask_file)
    
    image_rgbnir = np.stack((red,
                             green,
                             blue,
                             nir), axis=-1)
    
    image_rgb = np.dstack((red, green, blue))
          
    image_rgbnir = resize(image_rgbnir, (height, width), preserve_range=True, mode='symmetric')
    image_rgb = resize(image_rgb, (height, width), preserve_range=True, mode='symmetric')    
    mask = resize(mask, (height, width), preserve_range=True, mode='symmetric')

    
    mask /= 255
    image_rgbnir /= max_possible_input_value
    image_rgb /= max_possible_input_value
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgbnir)
    plt.title("RGB + N-Ir")
    plt.grid(False)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.title("RGB")
    plt.grid(False)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title("Ground Truth")
    plt.grid(False)
    plt.axis("off")
    
def show_rgb_image(img_files, mask_file, max_possible_input_value=65535):
    red = imread(img_files[0])
    green = imread(img_files[1])
    blue = imread(img_files[2])
    nir = imread(img_files[3])
    
    red = red / max_possible_input_value
    green = green / max_possible_input_value
    blue = blue / max_possible_input_value
    nir = nir / max_possible_input_value
    
    mask = imread(mask_file)
    mask = mask / 255
    
    image_rgbnir = (np.dstack((red,green,blue,nir)) * 255.999).astype(np.uint8)
    image_rgb = (np.dstack((red,green,blue)) * 255.999).astype(np.uint8)
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgbnir)
    plt.title("RGB + N-Ir")
    plt.grid(False)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.title("RGB")
    plt.grid(False)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title("Ground Truth")
    plt.grid(False)
    plt.axis("off")
    