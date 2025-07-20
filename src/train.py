# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:19:26 2024

@author: ammar
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from experiment_models.cloud_net import model_arch
# from experiment_models.AtrousCloudNet import build_model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from src.generators_gs import train_batch_generator, validation_batch_generator
from sklearn.model_selection import train_test_split
from src.utils import ADAMLearningRateTracker
from src.utils import get_input_image
from src.utils import jacc_coef
import matplotlib.pyplot as plt


# training number
TRAIN_NO = '-002'

# set all random seeds of the program (Python, NumPy, and TensorFlow)
os.environ["PYTHONHASHSEED"] = str(42)
set_random_seed(42)

GLOBAL_PATH = 'D:\\Projects\\cloud-segmentation\\'
DATASET_PATH = 'D:\\Datasets\\Cloud_Images_Dataset\\'
TRAIN_DIR = os.path.join(DATASET_PATH, 'Cloud_training')
TEST_DIR = os.path.join(DATASET_PATH, 'Cloud_test')

width = height = 192
num_channels = 1
num_clasess = 1
num_epochs = 42
lr_start = 1e-4
lr_end = 1e-8
val_ratio = 0.2
patience = 10
decay_factor = 0.7
batch_size = 8
max_bit = 65535  # maximum gray level in landsat 8 images
experiment_name = "Cloud-Net-GS-epoch-50" # + str(num_epochs)
weights_path = os.path.join(GLOBAL_PATH, 'saved_model_weights', experiment_name + '.h5')
train_resume = True
prev_weight = os.path.join(GLOBAL_PATH, 'saved_model_weights', 'Cloud-Net-GS-epoch-8.h5')

file_train_images = pd.read_csv(os.path.join(TRAIN_DIR, 'training_patches_95-cloud_nonempty.csv'))
train_images, train_masks = get_input_image(file_train_images, TRAIN_DIR, is_train=True)

def train():
    model = model_arch(input_rows=height,
                        input_cols=width,
                        num_of_channels=num_channels,
                        num_of_classes=num_clasess)
    
    model.compile(optimizer=Adam(learning_rate=lr_start), 
                  loss=jacc_coef, 
                  metrics=[jacc_coef])
    # model.summary()
    
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=lr_end, verbose=1)
    csv_logger = CSVLogger(os.path.join(GLOBAL_PATH, 'logger', 'LOG_' + experiment_name + TRAIN_NO + '.log'))
    
    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_images, train_masks,
                                                                                      test_size=val_ratio,
                                                                                      random_state=42, shuffle=True)
    
    if train_resume:
        model.load_weights(prev_weight)
        print("\nTraining resumed...")
    else:
        print("\nTraining started from scratch... ")
        
    print("Experiment name: ", experiment_name)
    print("Input image size: ", (width, height))
    print("Number of input spectral bands: ", num_channels)
    print("Learning rate: ", lr_start)
    print("Batch size: ", batch_size, "\n")
    
    history = model.fit_generator(
                generator=train_batch_generator(list(zip(train_img_split, train_msk_split)), height, width, batch_size, max_bit),
                steps_per_epoch=np.ceil(len(train_img_split) / batch_size), epochs=num_epochs, verbose=1,
                validation_data=validation_batch_generator(list(zip(val_img_split, val_msk_split)), height, width, batch_size, max_bit),
                validation_steps=np.ceil(len(val_img_split) / batch_size),
                callbacks=[model_checkpoint, lr_reducer, ADAMLearningRateTracker(lr_end), csv_logger])
    
    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.legend()

if __name__ == '__main__':
    with tf.device('/device:GPU:0'):
        train()

