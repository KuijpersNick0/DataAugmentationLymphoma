# --- create new images ---

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import pandas
from scipy.io import loadmat 
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time 
import math
import random
from scipy.fftpack import dct, idct # import for discrete cosine transform
from torchsummary import summary 
# path for the functions of DCT scripts. 

import method_1_DCT as method_1
import method_2_DCT_variant as method_2
import method_3_DCT_variant as method_3
from PIL import Image

# --- for device with cuda environment ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Load dataset ---
matPath = ".../DatasColor_29.mat"
data = scipy.io.loadmat(matPath) 

DIV = data['DATA'][0][2] # Division between training and test set
DIM1 = 299  # Number of training patterns
DIM2 = 374 # Number of patterns
NF = 5 
yE = data['DATA'][0][1]  # Labels of the patterns 
Images = data["DATA"][0][0][0] # Images

# ------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------
def create_new_images_method_1(Images, directory_path): # every component of the feature vector is randomly set to zero with a given probability
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("\n-----------------------------------------------------")
        print(f"Directory '{directory_path}' created successfully.")
        print("New images will be create using method_1")
        print("-----------------------------------------------------")

        # for each image of the list of images create a new image and save it into the folder
        for i, img in enumerate(Images): 
            new_img = method_1.method1_DCT(img) #create the new image
            image_to_save = Image.fromarray(new_img)
            image_to_save.save(f"{directory_path}/new_img_{i}.png")    
    
    else:
        print("\n-----------------------------------------------------")
        print(f"Directory '{directory_path}' already exists.")
        print("New images will be putted inside using method_1")
        print("-----------------------------------------------------")
                # for each image of the list of images create a new image and save it into the folder
        for i, img in enumerate(Images): 
            new_img = method_1.method1_DCT(img) #create the new image
            image_to_save = Image.fromarray(new_img)
            image_to_save.save(f"{directory_path}/new_img_{i}.png")    

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def create_new_images_method_2(Images, directory_path): # some of the features at a random value extracted from a Gaussian distribution are reset
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("\n-----------------------------------------------------")
        print(f"Directory '{directory_path}' created successfully.")
        print("New images will be create using method_2")
        print("-----------------------------------------------------")
        # for each image of the list of images create a new image and save it into the folder
        for i, img in enumerate(Images): 
            new_img = method_2.method2dct(img) #create the new image
            image_to_save = Image.fromarray(new_img)
            image_to_save.save(f"{directory_path}/new_img_{i}.png")    
    
    else:
        print("\n-----------------------------------------------------")
        print(f"Directory '{directory_path}' already exists.")
        print("New images will be putted inside using method_2")
        print("-----------------------------------------------------")
        for i, img in enumerate(Images): 
            new_img = method_2.method2dct(img) #create the new image
            image_to_save = Image.fromarray(new_img)
            image_to_save.save(f"{directory_path}/new_img_{i}.png")    
        
# ------------------------------------------------------------------------------------------------------------------------------------------------------
def create_new_images_method_3(Images, directory_path): # five random images in the dataset are selected that have the same label as a given image
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("\n-----------------------------------------------------")
        print(f"Directory '{directory_path}' created successfully.")
        print("New images will be create using method_3")
        print("-----------------------------------------------------")
        for i, img in enumerate(Images): 
            new_img = method_3.method3dct(img) #create the new image
            image_to_save = Image.fromarray(new_img)
            image_to_save.save(f"{directory_path}/new_img_{i}.png")    
    
    else:
        print("\n-----------------------------------------------------")
        print(f"Directory '{directory_path}' already exists.")
        print("New images will be putted inside using method_3")
        print("-----------------------------------------------------")
        for i, img in enumerate(Images): 
            new_img = method_3.method3dct(img) #create the new image
            image_to_save = Image.fromarray(new_img)
            image_to_save.save(f"{directory_path}/new_img_{i}.png")    
            
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Function calling: 
#Create a small batch of images (to simulate the training set.)
first_30_image = list(Images[:30])
directory_name_m1 = '.../Method_1_images_data_augment'
directory_name_m2 = '.../Method_2_images_data_augment'
directory_name_m3 = '.../Method_3_images_data_augment'
# For each ome of the method create the new images: 
# --- method_1 ---
create_new_images_method_1(first_30_image, directory_name_m1)
# --- method_2 ---
create_new_images_method_2(first_30_image, directory_name_m2)
# --- method 3 --- 
create_new_images_method_3(first_30_image, directory_name_m3)