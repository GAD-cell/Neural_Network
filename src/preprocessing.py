import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import math
from sklearn.utils import shuffle

"""
This module contains functions for preprocessing image data for neural network training.
It includes functions for cropping, resizing, and preparing images for training.
"""

IMG_SIZE = 200

def crop_image_from_gray(img, tol=7):
    """
    Crop image based on grayscale values.
    
    Args:
        img (numpy.ndarray): Input image
        tol (int): Tolerance value for cropping
        
    Returns:
        numpy.ndarray: Cropped image
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop(path, sigmaX=50):
    """
    Crop image in a circular shape.
    
    Args:
        path (str): Path to the image file
        sigmaX (int): Sigma value for Gaussian blur
        
    Returns:
        numpy.ndarray: Processed image
    """
    img = cv2.imread(path)
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x, y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img 

def load_data_train(n, size):
    """
    Load training data labels.
    
    Args:
        n (int): Number of samples to load
        size (int): Size of the images
        
    Returns:
        numpy.ndarray: One-hot encoded labels
    """
    test_label_o = np.zeros((n, 5, 1))
    
    # Update the path to use a relative path or environment variable
    csv_path = os.path.join(os.getenv('DATA_DIR', '.'), 'training_labels.csv')
    
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = np.array(list(csv_reader))
    
    test_label = np.reshape((rows[:, 1])[1:n+1], (n, 1))

    for i in range(n):
        v = [[0], [0], [0], [0], [0]]
        v[int(test_label[i][0])] = [1]
        test_label_o[i] = np.array(v)
    
    return test_label_o

def process_training_images(data_dir=None):
    """
    Process training images and save them as numpy arrays.
    
    Args:
        data_dir (str, optional): Directory containing the images
    """
    # Use environment variable or default to current directory
    if data_dir is None:
        data_dir = os.getenv('DATA_DIR', '.')
    
    test_images = np.zeros((400, IMG_SIZE, IMG_SIZE))
    
    for i in range(1, 401):
        print(i, "processing...")
        # Update the path to use a relative path or environment variable
        image_path = os.path.join(data_dir, f"IDRiD_{str(i).zfill(3)}.jpg")
        image = circle_crop(image_path)
        test_images[i-1] = image[:, :, 1]  # Use green channel for retinal images
    
    np.save('train_image', test_images)
    np.save('train_labels', load_data_train(400, 200))
    
    print("Processing complete. Files saved as 'train_image.npy' and 'train_labels.npy'")