from textwrap import wrap
import os
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from tensorflow import keras
# import tensorflow_addons as tfa
# from keras_cv_attention_models import Classifiers
import torch, torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import scipy.io     
from scipy.io import loadmat   
import torch.nn as nn
import torch.optim as optim 
import time 
import math
import random
from scipy.fftpack import dct, idct # import for discrete cosine transform
from torchsummary import summary 
from PIL import Image

# Constants for text encoder
PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77

# Load the tokenizer
tokenizer = SimpleTokenizer()

# Text encoder model
text_encoder = TextEncoder(MAX_PROMPT_LENGTH)

# Import data, as in main.py
matPath = "../data/DatasColor_29.mat"
class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, mat_path, transform=None, train=True, fold=1):
        self.mat_data = scipy.io.loadmat(mat_path)
        self.fold = fold
        self.train = train
        self.transform = transform
        self.train_indices = self.mat_data['DATA'][0][2][fold-1, :299] - 1
        self.test_indices = self.mat_data['DATA'][0][2][fold-1, 299:374] - 1
        self.y_train = self.mat_data['DATA'][0][1][0, self.train_indices]
        self.y_test = self.mat_data['DATA'][0][1][0, self.test_indices]
        self.num_classes = len(np.unique(self.y_train))
        self.images = self.mat_data["DATA"][0][0][0]   # contains 374 images with images being of size (227,227,3)
        
    def __len__(self):
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)
        
    def __getitem__(self, idx):
        if self.train:
            img = Image.fromarray(self.images[self.train_indices[idx]])
            label = self.y_train[idx] - 1  # shift the labels to start from 0
        else:
            img = Image.fromarray(self.images[self.test_indices[idx]])
            label = self.y_test[idx] - 1   # shift the labels to start from 0
            
        if self.transform:
            img = self.transform(img)
        
        # Map labels to captions
        if label == 0:
            caption = "microscopic image of chronic lymphocytic leukemia"
        elif label == 1:
            caption = "microscopic image of follicular lymphoma"
        elif label == 2:
            caption = "microscopic image of mantle cell lymphoma"
        else:
            raise ValueError(f"Invalid label: {label}")

         # Tokenize and pad the caption
        tokens = tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        tokenized_caption = np.array(tokens)

        return img, tokenized_caption

# # Neural network parameters
miniBatchSize = 4
num_classes = 3

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=227, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MyDataset(mat_path=matPath, transform=train_transforms, train=True, fold=1)
# valid_dataset = MyDataset(mat_path=matPath, transform=val_transforms, train=False, fold=1)

train_data_loader = DataLoader(train_dataset, batch_size=miniBatchSize, shuffle=True)
# valid_data_loader = DataLoader(valid_dataset, batch_size=miniBatchSize, shuffle=False)


# Define the image augmentation and preprocessing
RESOLUTION = 256
augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.CenterCrop(RESOLUTION, RESOLUTION),
        keras_cv.layers.RandomFlip(),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)



# Prepare the data for training
image_paths = []
tokenized_captions = []
for item in train_data_loader.dataset:
    image_paths.append(item[0].numpy())
    tokenized_captions.append(item[1])

image_batch = np.array(image_paths)
caption_batch = np.array(tokenized_captions)

# Apply image augmentation and encoding
augmented_images = augmenter(image_batch)
encoded_captions = text_encoder([caption_batch, POS_IDS], training=False)

# Prepare the data dictionary
data_dict = {
    "images": augmented_images,
    "tokens": caption_batch,
    "encoded_text": encoded_captions
}

# Convert the dictionary to TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(data_dict)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Example usage: Iterate over the dataset
for batch in dataset:
    images = batch["images"]
    tokens = batch["tokens"]
    encoded_text = batch["encoded_text"]
    # Perform further processing or training with the batch
