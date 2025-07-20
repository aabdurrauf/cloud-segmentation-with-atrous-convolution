# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:50:37 2024

@author: ammar
"""

"""
how to test your model:
   1. import the model
   2. import the test batch generator. make sure you import the right
      generator with the right number of channels (i.e. RGB, RGB-Nir, or GS)
   3. change the num_channels variable according to your input data
   4. change the experiment_name as your saved model weight file name (without ".h")
   5. if your computer RAM memory cannot handle huge data, you can split the
      test into two phases.  (open one of the .csv file names: 01 or 02)

""" 

# import the model
from experiment_models.cloud_net import model_arch
# from experiment_models.AtrousCloudNet import build_model

# import the generator according to the input channel type
from src.generators_gs import test_batch_generator
from src.generators_gs import test_batch_generator_red
from src.generators_gs import test_batch_generator_green
from src.generators_gs import test_batch_generator_blue
from src.generators_gs import test_batch_generator_nir
# from src.generators_rgb import test_batch_generator

from src.utils import get_input_image
import tifffile as tiff
import pandas as pd
import numpy as np
import os


width = height = 384
num_channels = 1
num_clasess = 1
batch_size = 1
max_bit = 65535

GLOBAL_PATH = 'D:\\Projects\\cloud-segmentation\\'
DATASET_PATH = 'D:\\Datasets\\Cloud_Images_Dataset\\'
TEST_DIR = os.path.join(DATASET_PATH, 'Cloud_test')
PRED_DIR = os.path.join(DATASET_PATH, 'Cloud_prediction')

COLOR_CHANNEL = '_red'

experiment_name = 'Cloud-Net-GS-epoch-50'
# experiment_name = 'AtrousCloud-Net-GS-epoch-50'
# experiment_name = 'Cloud-Net-RGB-epoch-50-latest'
# experiment_name = 'AtrousCloud-Net-RGB-epoch-50-latest'

weights_path = os.path.join(GLOBAL_PATH, 'saved_model_weights', experiment_name + '.h5')

# getting input images names
# test_patches_csv_name = 'test_patches_38-Cloud_01.csv'
test_patches_csv_name = 'test_patches_38-Cloud_02.csv'

df_test_img = pd.read_csv(os.path.join(TEST_DIR, test_patches_csv_name))
test_img, test_ids = get_input_image(df_test_img, TEST_DIR, is_train=False)




# def predict_mask():
model = model_arch(input_rows=height,
                    input_cols=width,
                    num_of_channels=num_channels,
                    num_of_classes=num_clasess)

model.load_weights(weights_path)

print("\nExperiment name: ", experiment_name)
print("Prediction started... ")
print("Input image size = ", (height, width))
print("Number of input spectral bands = ", num_channels)
print("Batch size = ", batch_size)

imgs_mask_test = model.predict_generator(
                            generator=test_batch_generator_red(test_img, height, width, batch_size, max_bit),
                            steps=np.ceil(len(test_img) / batch_size))

print("Saving predicted cloud masks on disk... \n")

pred_folder = experiment_name + '_preddicted_masks' + COLOR_CHANNEL
if not os.path.exists(os.path.join(PRED_DIR, pred_folder)):
    os.mkdir(os.path.join(PRED_DIR, pred_folder))

for image, image_id in zip(imgs_mask_test, test_ids):
    image = (image[:, :, 0]).astype(np.float32)
    tiff.imsave(os.path.join(PRED_DIR, pred_folder, str(image_id)), image)

    