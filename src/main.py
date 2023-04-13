import scipy.io
import sklearn
from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


matPath = "../data/DatasColor_29.mat"
 

def showMatFile(matPath):
    # Load the mat file
    images = scipy.io.loadmat(matPath)  
    print(images.keys())  
    print(images['DATA'].shape)

    print(images['DATA'][0][0][0].shape)
    print(images['DATA'][0][0][0][0].shape)

    # print(images['DATA'][0][0][0][0])  

    # Montre image après image
    for i in range(0, 10):
        img = images['DATA'][0][0][0][i]
        plt.imshow(img)
        plt.show()



showMatFile(matPath)