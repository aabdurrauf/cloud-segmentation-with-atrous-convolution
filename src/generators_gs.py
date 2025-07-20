# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:11:10 2024

@author: ammar

source: https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection/blob/master/Cloud-Net/generators.py
"""

from src.augmentation import flip_img_and_msk, rotate_cclk_img_and_msk
from src.augmentation import rotate_clk_img_and_msk, zoom_img_and_msk
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import random

np.random.seed(42)
random.seed(42)

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
            
            mask = imread(mask)
            
            image = np.stack((image_red,
                              image_green,
                              image_blue), axis=-1)
            
            image = rgb2gray(image)
            
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
            
            mask = imread(mask)
            
            image = np.stack((image_red,
                              image_green,
                              image_blue), axis=-1)
            
            image = rgb2gray(image)
            
            image = resize(image, (height, width), preserve_range=True, mode='symmetric')
            mask = resize(mask, (height, width), preserve_range=True, mode='symmetric')
            
            mask = mask[..., np.newaxis] # adding new dimension
            mask /= 255
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

            image = np.stack((image_red, image_green, image_blue), axis=-1)
            image = rgb2gray(image)

            image = resize (image, (height, width), preserve_range=True, mode='symmetric')


            image_list.append(image)

        counter += 1
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0
    
def test_batch_generator_red(test_files, 
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

            image = np.stack((image_red), axis=-1)

            image = resize (image, (height, width), preserve_range=True, mode='symmetric')
            
            image /= max_possible_input_value
            image_list.append(image)

        counter += 1
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0

def test_batch_generator_green(test_files, 
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

            image_green = imread(file[1])

            image = np.stack((image_green), axis=-1)

            image = resize (image, (height, width), preserve_range=True, mode='symmetric')
            
            image /= max_possible_input_value
            image_list.append(image)

        counter += 1
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0

def test_batch_generator_blue(test_files, 
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

            image_blue = imread(file[2])

            image = np.stack((image_blue), axis=-1)

            image = resize (image, (height, width), preserve_range=True, mode='symmetric')
            
            image /= max_possible_input_value
            image_list.append(image)

        counter += 1
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0

def test_batch_generator_nir(test_files, 
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

            image_nir = imread(file[3])

            image = np.stack((image_nir), axis=-1)

            image = resize (image, (height, width), preserve_range=True, mode='symmetric')
            
            image /= max_possible_input_value
            image_list.append(image)

        counter += 1
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0
    