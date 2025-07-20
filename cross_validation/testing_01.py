# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:04:50 2024

@author: ammar
"""
# In[1]:
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Nadam, RMSprop, SGD
from cross_validation.utils import get_input_image, jacc_coef
from cross_validation.utils import train_batch_generator, validation_batch_generator, test_batch_generator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tifffile as tiff
import pandas as pd
import numpy as np
import os


# import your model here
from experiment_models.cloud_net import cloud_net, model_name

# In[2]:
# experiment variables
experiment_name = model_name

# CONSTANTS
GLOBAL_PATH = 'D:\\Projects\\cloud-segmentation\\'
DATASET_PATH = 'D:\\Datasets\\Cloud_Images_Dataset\\'
TRAIN_DIR = os.path.join(DATASET_PATH, '38-Cloud_training')
TEST_DIR = os.path.join(DATASET_PATH, '38-Cloud_test')
width = height = 192
num_channels = 4
batch_size = 8
K_fold = 5
max_bit = 65535
num_epochs = 100

# In[3]:
# hyperparameters
optimizers = [Adam, Adagrad, Adamax, Nadam, RMSprop, SGD]
learning_rate = [1e-4]
lr_end = 1e-8
activation_functions = ['relu', 'tanh', 'leaky_relu']
loss = [jacc_coef, 'binary_crossentropy']

# In[4]:
# get the file list
file_train_images = pd.read_csv(os.path.join(TRAIN_DIR, 'training_patches_38-Cloud.csv'))
train_images, train_masks = get_input_image(file_train_images, TRAIN_DIR, is_train=True)
group_size = len(train_images) // K_fold


# seperate 
image_groups = []
mask_groups = []
for i in range(K_fold):
    image_temp = train_images[i * group_size:group_size * (i + 1)]
    image_groups.append(image_temp)
    
    mask_temp = train_masks[i * group_size:group_size * (i + 1)]
    mask_groups.append(mask_temp)

opt = Adam

# In[5]:
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
i = 0
train_resume = True
with tf.device('/device:GPU:0'):
    # edit your model here with parameterized activation function
    model = cloud_net
    
    model.compile(optimizer=opt(learning_rate=learning_rate[0]), 
                  loss=loss[0], 
                  metrics=[loss[0]])
    
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
    if train_resume:
        model.load_weights(os.path.join(GLOBAL_PATH, 'saved_model_weights', 'New-Cloud-Net-002-500-epochs.h5'))
        print("\nTraining resumed...")
    
    model.fit_generator(
        generator=train_batch_generator(list(zip(train_img_split, train_msk_split)), height, width, batch_size, max_bit),
        steps_per_epoch=np.ceil(len(train_img_split) / batch_size), epochs=num_epochs, verbose=1,
        validation_data=validation_batch_generator(list(zip(val_img_split, val_msk_split)), height, width, batch_size, max_bit),
        validation_steps=np.ceil(len(val_img_split) / batch_size),
        callbacks=[lr_reducer])
    
# In[10]:
    
model.save_weights('D://Projects//cloud-segmentation//saved_model_weights//New-Cloud-Net-002-500-epochs.h5')
# In[7]:

import random
import matplotlib.pyplot as plt    

# evaluate the model
predicted_mask = model.predict_generator(generator=test_batch_generator(test_images, height, width, batch_size, max_bit), steps=np.ceil(len(test_images) / batch_size))

# In[8]:
# n = random.randint(0, len(predicted_mask))
n = 159

# image
plt.subplot(2, 2, 1)

image_red = imread(test_images[n][0])
image_green = imread(test_images[n][1])
image_blue = imread(test_images[n][2])
# image_nir = imread(test_images[n][3])
image = np.stack((image_red,
                  image_green,
                  image_blue), axis=-1)
                  # image_nir), axis=-1)
image = resize(image, (height, width), preserve_range=True, mode='symmetric')
image /= 65535

plt.imshow(image)
plt.title('image')
plt.grid(False)

# ground truth
plt.subplot(2,2,2)

mask = imread(test_masks[n])
mask = resize(mask, (height, width), preserve_range=True, mode='symmetric')
mask /= 255
mask_thres = [[1 if x >= 0.5 else 0 for x in m] for m in mask]
plt.imshow(mask_thres)
plt.title('ground truth')
plt.grid(False)

# predicted mask
plt.subplot(2,2,3)
pred_mask = predicted_mask[n]
pred_mask = resize(pred_mask, (height, width), preserve_range=True, mode='symmetric')
plt.imshow(pred_mask)
plt.title('predicted')
plt.grid(False)

# predicted binary mask
plt.subplot(2,2,4)
thres_val = np.mean(pred_mask)
# thres_val = 0.7
pred_thres = [[1 if x >= thres_val else 0 for x in m] for m in pred_mask]
plt.imshow(pred_thres, cmap='gray')
plt.title('pred. mask > 0.5 tresh')
plt.rcParams.update({'font.size': 6})
plt.grid(False)
    
## In[9]:
smooth = 0.0000001

def compute_intersection(x, y):
    # total = 0
    # for ix in range(len(x)):
    #     total += (x[ix] * y[ix])

    # print(total)
    # return total
    return np.matmul(x, y)

def compute_union(x, y):
    total = 0
    for ix in range(len(x)):
        if x[ix] == 1 or y[ix] == 1:
            total += 1

    return total

        
def iou_func(y_true, y_pred):
    mask_flatten = np.array(y_true).flatten()
    pred_flatten = np.array(y_pred).flatten()
    
    intersection = compute_intersection(mask_flatten, pred_flatten)
    print(f'intersection: {intersection}')
    union = compute_union(mask_flatten, pred_flatten)
    print(f'union: {union}')
    
    return (intersection + smooth)/(union + smooth)

IoU = iou_func(mask_thres, pred_thres)
print(f'{n}: {IoU}')

# In[11]:
from src.metrics import iou_func, iou_score, precision_score, recall_score, specificity_score, accuracy_score
# IoU = iou_func(mask_thres, pred_thres)
# print(f'{n} iou_func: {IoU}')


iou = iou_score(mask_thres, pred_thres)
precision = precision_score(mask_thres, pred_thres)
recall = recall_score(mask_thres, pred_thres)
specificity = specificity_score(mask_thres, pred_thres)
accuracy = accuracy_score(mask_thres, pred_thres)
print(f'IoU: {iou}\nprecision: {precision}\nrecall: {recall}' + 
      f'\nspecificity: {specificity}\naccuracy: {accuracy}\n')

# In[10]:
from src.metrics import iou_func, iou_score
scores = []
index = 0
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
    
    # gt = gt.flatten()
    # predicted = predicted.flatten()
    
    # acc_value = accuracy_score(gt, predicted)
    # f1_score_value = f1_score(gt, predicted, labels=[0, 1], average='binary')
    # iou_score = jaccard_score(gt, predicted, labels=[0, 1], average='binary')
    # recall_value = recall_score(gt, predicted, labels=[0, 1], average='binary')
    # precision_value = precision_score(gt, predicted, labels=[0, 1], average='binary')
    
    iou = iou_score(gt, predicted)
    precision = precision_score(gt, predicted)
    recall = recall_score(gt, predicted)
    specificity = specificity_score(gt, predicted)
    accuracy = accuracy_score(gt, predicted)
    print(f'index: {index}\nIoU: {iou}\nprecision: {precision}\nrecall: {recall}' + 
          f'\nspecificity: {specificity}\naccuracy: {accuracy}\n')
    
    index += 1
    
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
print(f"{i}. IoU mean:", mean_score[0])
print(f"{i}. Precision:", mean_score[1])
print(f"{i}. Recall:", mean_score[2])
print(f"{i}. Specificity:", mean_score[3])
print(f"{i}. Overall Accuracy:", mean_score[4], '\n')


