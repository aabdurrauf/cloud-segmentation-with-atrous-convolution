# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:10:43 2024

@author: ammar
"""

"""
this file is intended for experimenting with the models to get the model
with the best accuracy using 4-Fold Cross Validation method.
the steps are as follows:
    1. 840 image and mask samples are read from the dataset
    2. the proposed model will be trained with the followiwng
       cross validation rule:
        - split the dataset into 4 groups where each group contains 168 samples
        - the proposed model will be trained 4 times using 4 groups and then 
          tested using the remaining group
        - the result will be saved in an excel file for evaluation
    3. easily tuned hyperparameters are:
        - learning rate
        - activation function
        - loss function
        - optimizer
"""
# In[1]:
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Nadam, RMSprop, SGD
from cross_validation.utils import get_input_image, jacc_coef
# from cross_validation.utils import train_batch_generator, validation_batch_generator, test_batch_generator
from cross_validation.utils import train_batch_generator_rgb, validation_batch_generator_rgb, test_batch_generator_rgb
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import set_random_seed
from sklearn.model_selection import train_test_split
# import tifffile as tiff
import pandas as pd
import numpy as np
import os


# import your model here
# from experiment_models.cloud_net import model_arch, model_name
# from experiment_models.my_model_02 import build_model, model_name
from experiment_models.my_model_09 import build_model, model_name


# In[2]:
# experiment variables
experiment_name = model_name

# CONSTANTS
GLOBAL_PATH = 'D:\\Projects\\cloud-segmentation\\'
DATASET_PATH = 'D:\\Datasets\\Cloud_Images_Dataset\\'
TRAIN_DIR = os.path.join(DATASET_PATH, '38-Cloud_training')
TEST_DIR = os.path.join(DATASET_PATH, '38-Cloud_test')
weights_path = os.path.join(GLOBAL_PATH, 'saved_model_weights', experiment_name + '.h5')

width = height = 192
num_channels = 3
batch_size = 4
skip_sample = 20
K_fold = 4
max_bit = 65535
num_epochs = 40

# set all random seeds of the program (Python, NumPy, and TensorFlow)
os.environ["PYTHONHASHSEED"] = str(42)
set_random_seed(42)

# In[3]:
# hyperparameters
# optimizers = [Adam, Adagrad, Adamax, Nadam, RMSprop]
optimizers = [Adam]
# optimizers_name = ['Adam', 'Adagrad', 'Adamax', 'Nadam']
optimizers_name = ['Adam']
learning_rate = 1e-4
lr_end = 1e-8
momentum = 0.9
activation_functions = ['relu']

# In[4]:
# get the file list
file_train_images = pd.read_csv(os.path.join(TRAIN_DIR, 'training_patches_38-Cloud.csv'))
train_images, train_masks = get_input_image(file_train_images, TRAIN_DIR, skip_sample=skip_sample, is_train=True)
group_size = len(train_images) // K_fold


# seperate 
image_groups = []
mask_groups = []
for i in range(K_fold):
    image_temp = train_images[i * group_size:group_size * (i + 1)]
    image_groups.append(image_temp)
    
    mask_temp = train_masks[i * group_size:group_size * (i + 1)]
    mask_groups.append(mask_temp)
# In[5]:
from src.metrics import iou_score, precision_score, recall_score, specificity_score, accuracy_score
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
import matplotlib.pyplot as plt

with tf.device('/device:GPU:0'):
    for opt_name, opt in zip(optimizers_name, optimizers):
        for af in activation_functions:
            for i in range(K_fold):
                # clear previous session
                tf.keras.backend.clear_session()
                
                # edit your model here with parameterized activation function
                model = build_model(num_of_channels=num_channels)
                model.compile(optimizer=opt(learning_rate=learning_rate), 
                              loss=jacc_coef, 
                              metrics=[jacc_coef])
                
                model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
                lr_reducer = ReduceLROnPlateau(factor=0.7, cooldown=0, patience=8, min_lr=lr_end, verbose=1)
                
                # take 4 groups for train and keep 1 for test
                test_images = image_groups[i]
                test_masks = mask_groups[i]
                
                train_images_fold = []
                train_masks_fold = []
                for j in range(K_fold):
                    if j != i:
                        for img in image_groups[j]:
                            train_images_fold.append(img)
                        for msk in mask_groups[j]:
                            train_masks_fold.append(msk)
                
                train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_images_fold, 
                                                                                                  train_masks_fold,
                                                                                                  test_size=0.2,
                                                                                                  random_state=42, 
                                                                                                  shuffle=True)
                
                history = model.fit_generator(
                    generator=train_batch_generator_rgb(list(zip(train_img_split, train_msk_split)), height, width, batch_size, max_bit),
                    steps_per_epoch=np.ceil(len(train_img_split) / batch_size), epochs=num_epochs, verbose=1,
                    validation_data=validation_batch_generator_rgb(list(zip(val_img_split, val_msk_split)), height, width, batch_size, max_bit),
                    validation_steps=np.ceil(len(val_img_split) / batch_size),
                    callbacks=[model_checkpoint, lr_reducer])
                
                plt.figure()
                plt.plot(history.history['loss'],label='loss')
                plt.plot(history.history['val_loss'],label='val_loss')
                plt.legend()
                
                # evaluate the model
                model.load_weights(weights_path)
                predicted_mask = model.predict_generator(generator=test_batch_generator_rgb(test_images, height, width, batch_size, max_bit),
                                                 steps=np.ceil(len(test_images) / batch_size))
                
                scores = []
                for gt_file, predicted in zip(test_masks, predicted_mask):
                    gt = imread(gt_file)
                    gt = resize(gt, (height, width), preserve_range=True, mode='symmetric')
                    gt = gt > 0.5
                    gt = gt.astype(np.int32)
                    gt = np.reshape(gt, (height, width))
                    
                    # potential research on how to determine the best threshold value
                    predicted = predicted > 0.5
                    predicted = predicted.astype(np.int32)
                    predicted = np.reshape(predicted, (height, width))
                    
                    iou = iou_score(gt, predicted)
                    precision = precision_score(gt, predicted)
                    recall = recall_score(gt, predicted)
                    specificity = specificity_score(gt, predicted)
                    accuracy = accuracy_score(gt, predicted)
                    
                    scores.append([iou, precision, recall, specificity, accuracy])
                    
                    
                # pred_dir = experiment_name + '-' + str(1)
                # if not os.path.exists(os.path.join('cross_validation', 'predicted masks')):
                #     os.mkdir(os.path.join('cross_validation', 'predicted masks'))
                
                # for image, image_id in zip(images_mask_test, test_ids):
                #     image = (image[:, :, 0]).astype(np.float32)
                #     tiff.imsave(os.path.join('cross_validation', 'predicted masks', str(image_id)), image)

                mean_score = np.mean(scores, axis=0)
                
                print("Experiment name: ", experiment_name)
                print(f"{i}. --- Results ---")
                # print(f"{i}. Activation:", af)
                print(f"{i}. IoU mean:", mean_score[0])
                print(f"{i}. Precision:", mean_score[1])
                print(f"{i}. Recall:", mean_score[2])
                print(f"{i}. Specificity:", mean_score[3])
                print(f"{i}. Overall Accuracy:", mean_score[4], '\n')
                
                with open(GLOBAL_PATH + 'cross-validation_' + model_name + '.txt', 'a') as file:
                    file.write(f"Experiment name: {experiment_name}\n")
                    # file.write(f"Optimizer: {opt_name}\n")
                    # file.write(f"Activation: {af}\n")
                    file.write(f"{i}. --- Results ---\n")
                    file.write(f"{i}. IoU mean: {mean_score[0]}\n")
                    file.write(f"{i}. Precision: {mean_score[1]}\n")
                    file.write(f"{i}. Recall: {mean_score[2]}\n")
                    file.write(f"{i}. Specificity: {mean_score[3]}\n")
                    file.write(f"{i}. Overall Accuracy: {mean_score[4]}\n\n")
                
                print("Data has been written to cross-validation_" + model_name + ".txt")
        
        
