# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:31:26 2024

@author: ammar
"""

from src.augmentation import flip_img_and_msk, rotate_cclk_img_and_msk
from src.augmentation import rotate_clk_img_and_msk, zoom_img_and_msk
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from skimage.transform import resize
from skimage.io import imread
from tqdm import tqdm
import numpy as np
import random
import os


np.random.seed(42)
random.seed(42)

# get sample images
def get_input_image(name_list, name_dir, skip_sample=10, is_train=True):
    images = []
    masks = []
    test_ids = []
    
    i = 0
    for file_name in tqdm(name_list['name'], miniters=1000):
        red = 'red_' + file_name
        blue = 'blue_' + file_name
        green = 'green_' + file_name
        nir = 'nir_' + file_name
        
        if is_train:
            type_dir = 'train'
            image_files = []
            mask = 'gt_' + file_name
            mask_file = name_dir + '/train_gt/' + '{}.TIF'.format(mask)
            
        else:
            
            type_dir = 'test'
            image_files = []
            file_id = '{}.TIF'.format(file_name)
            test_ids.append(file_id)
            
        red_image = os.path.join(name_dir, type_dir + '_red', '{}.TIF'.format(red))
        green_image = os.path.join(name_dir, type_dir + '_green', '{}.TIF'.format(green))
        blue_image = os.path.join(name_dir, type_dir + '_blue', '{}.TIF'.format(blue))
        nir_image = os.path.join(name_dir, type_dir + '_nir', '{}.TIF'.format(nir))
        
        # this will only append every 10th sample image and mask
        if i % skip_sample == 0:
            image_files.append(red_image)
            image_files.append(green_image)
            image_files.append(blue_image)
            image_files.append(nir_image)
        
            images.append(image_files)
            masks.append(mask_file)
        
        i+=1
        
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

def train_batch_generator(zip_list, 
                          height, 
                          width, 
                          batch_size, 
                          shuffle=True, 
                          max_possible_input_value=65536):
    
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    
    counter = 0
    
    while True:
        if shuffle:
            random.shuffle(zip_list)
            
        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        images = []
        masks = []
        
        for img_files, mask in batch_files:
            image_red = imread(img_files[0])
            image_green = imread(img_files[1])
            image_blue = imread(img_files[2])
            image_nir = imread(img_files[3])
            
            mask = imread(mask)
            
            image = np.stack((image_red,
                              image_green,
                              image_blue,
                              image_nir), axis=-1)
            
            image = resize(image, (height, width), preserve_range=True, mode='symmetric')
            mask = resize(mask, (height, width), preserve_range=True, mode='symmetric')
            
            if np.random.randint(2, dtype=int) == 1:
                image, mask = flip_img_and_msk(image, mask)
                
            if np.random.randint(2, dtype=int) == 1:
                image, mask = rotate_clk_img_and_msk(image, mask)
            
            if np.random.randint(2, dtype=int) == 1:
                image, mask = rotate_cclk_img_and_msk(image, mask)
                
            if np.random.randint(2, dtype=int) == 1:
                image, mask = zoom_img_and_msk (image, mask)
                
            mask = mask[..., np.newaxis] # adding new dimension
            mask /= 255
            image /= max_possible_input_value
            images.append(image)
            masks.append(mask)
            
        counter += 1
        images = np.array(images)
        masks = np.array(masks)
        
        yield (images, masks)
        
        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0
            
def validation_batch_generator(zip_list, 
                               height, 
                               width, 
                               batch_size, 
                               shuffle=False, 
                               max_possible_input_value=65536):
    
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0
    
    while True:
        if shuffle:
            random.shuffle(zip_list)
            
        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        images = []
        masks = []
        
        for img_files, mask in batch_files:
            image_red = imread(img_files[0])
            image_green = imread(img_files[1])
            image_blue = imread(img_files[2])
            image_nir = imread(img_files[3])
            
            mask = imread(mask)
            
            image = np.stack((image_red,
                              image_green,
                              image_blue,
                              image_nir), axis=-1)
            
            image = resize(image, (height, width), preserve_range=True, mode='symmetric')
            mask = resize(mask, (height, width), preserve_range=True, mode='symmetric')
            
            mask = mask[..., np.newaxis] # adding new dimension
            mask /= 255
            image /= max_possible_input_value
            images.append(image)
            masks.append(mask)
        
        counter += 1
        images = np.array(images)
        masks = np.array(masks)
        
        yield (images, masks)
        
        if counter == number_of_batches:
            counter = 0
            
def test_batch_generator(test_files, 
                         height, 
                         width,
                         batch_size, 
                         max_possible_input_value=65536):
    
    number_of_batches = np.ceil(len(test_files) / batch_size)
    counter = 0

    while True:

        beg = batch_size * counter
        end = batch_size * (counter + 1)
        batch_files = test_files[beg:end]
        image_list = []

        for file in batch_files:

            image_red = imread(file[0])
            image_green = imread(file[1])
            image_blue = imread(file[2])
            image_nir = imread(file[3])

            image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)

            image = resize (image, (height, width), preserve_range=True, mode='symmetric')


            image /= max_possible_input_value
            image_list.append(image)

        counter += 1
        # print('counter = ', counter)
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0
            

def train_batch_generator_rgb(zip_list, 
                          height, 
                          width, 
                          batch_size, 
                          shuffle=True, 
                          max_possible_input_value=65536):
    
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    
    counter = 0
    
    while True:
        if shuffle:
            random.shuffle(zip_list)
            
        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        images = []
        masks = []
        
        for img_files, mask in batch_files:
            image_red = imread(img_files[0])
            image_green = imread(img_files[1])
            image_blue = imread(img_files[2])
            # image_nir = imread(img_files[3])
            
            mask = imread(mask)
            
            image = np.stack((image_red,
                              image_green,
                              image_blue), axis=-1)
            
            image = resize(image, (height, width), preserve_range=True, mode='symmetric')
            mask = resize(mask, (height, width), preserve_range=True, mode='symmetric')
            
            if np.random.randint(2, dtype=int) == 1:
                image, mask = flip_img_and_msk(image, mask)
                
            if np.random.randint(2, dtype=int) == 1:
                image, mask = rotate_clk_img_and_msk(image, mask)
            
            if np.random.randint(2, dtype=int) == 1:
                image, mask = rotate_cclk_img_and_msk(image, mask)
                
            if np.random.randint(2, dtype=int) == 1:
                image, mask = zoom_img_and_msk (image, mask)
                
            mask = mask[..., np.newaxis] # adding new dimension
            mask /= 255
            image /= max_possible_input_value
            images.append(image)
            masks.append(mask)
            
        counter += 1
        images = np.array(images)
        masks = np.array(masks)
        
        yield (images, masks)
        
        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0
            
def validation_batch_generator_rgb(zip_list, 
                               height, 
                               width, 
                               batch_size, 
                               shuffle=False, 
                               max_possible_input_value=65536):
    
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0
    
    while True:
        if shuffle:
            random.shuffle(zip_list)
            
        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        images = []
        masks = []
        
        for img_files, mask in batch_files:
            image_red = imread(img_files[0])
            image_green = imread(img_files[1])
            image_blue = imread(img_files[2])
            # image_nir = imread(img_files[3])
            
            mask = imread(mask)
            
            image = np.stack((image_red,
                              image_green,
                              image_blue), axis=-1)
            
            image = resize(image, (height, width), preserve_range=True, mode='symmetric')
            mask = resize(mask, (height, width), preserve_range=True, mode='symmetric')
            
            mask = mask[..., np.newaxis] # adding new dimension
            mask /= 255
            image /= max_possible_input_value
            images.append(image)
            masks.append(mask)
        
        counter += 1
        images = np.array(images)
        masks = np.array(masks)
        
        yield (images, masks)
        
        if counter == number_of_batches:
            counter = 0
            
def test_batch_generator_rgb(test_files, 
                         height, 
                         width,
                         batch_size, 
                         max_possible_input_value=65536):
    
    number_of_batches = np.ceil(len(test_files) / batch_size)
    counter = 0

    while True:

        beg = batch_size * counter
        end = batch_size * (counter + 1)
        batch_files = test_files[beg:end]
        image_list = []

        for file in batch_files:

            image_red = imread(file[0])
            image_green = imread(file[1])
            image_blue = imread(file[2])
            # image_nir = imread(file[3])

            image = np.stack((image_red, image_green, image_blue), axis=-1)

            image = resize (image, (height, width), preserve_range=True, mode='symmetric')


            image /= max_possible_input_value
            image_list.append(image)

        counter += 1
        # print('counter = ', counter)
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0