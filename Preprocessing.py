import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import math
from sklearn.utils import shuffle

IMG_SIZE = 200

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def circle_crop(path, sigmaX=50):   
  
    img = cv2.imread(path)
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img 

def load_data_train(n, size):
    test_label_o = np.zeros((n, 5,1))
    
    with open(r'C:\Users\Public\Disease Grading\Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = np.array(list(csv_reader))
    
    test_label = np.reshape((rows[:,1])[1:n+1], (n, 1))

    for i in range(n):
        #v = min(1, int(train_label[i][0]))
        v = [[0],[0],[0],[0],[0]]
        v[int(test_label[i][0])] = [1]
        test_label_o[i] = np.array(v)
    
    return(test_label_o)


#train_images = np.zeros((400, IMG_SIZE, IMG_SIZE))

#for i in range(1,401):
   # print(i, "training...")
  #  path = r"C:\Users\Public\Disease Grading\Original Images\Training Set\IDRiD_"+(2-int(math.log10(i)))*"0"+str(i)+".jpg"
  #  image = circle_crop(path)
  #  train_images[i-1] = image[:,:,1]

#train_shuffle , trainlabel_shuffle = shuffle(train_images , load_data_train(400,200) , random_state = 77)



test_images = np.zeros((400, IMG_SIZE, IMG_SIZE))

for i in range(1,401):
    print(i, "testing...")
    path = r"C:\Users\Public\Disease Grading\Original Images\Training Set\IDRiD_"+(2-int(math.log10(i)))*"0"+str(i)+".jpg"
    image = circle_crop(path)
    test_images[i-1] = image[:,:,1]


np.save('train_image' , test_images )
np.save('train_labels' , load_data_train(400, 200))
