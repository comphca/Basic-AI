import numpy as np
import pandas as pd
import os
from PIL import Image

def load_Img(imgDir,imgFoldName):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = np.empty((imgNum,1,12,12),dtype="float32")
    label = np.empty((imgNum,),dtype="uint8")
    for i in range (imgNum):
        img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data = arr
        label[i] = int(imgs[i].split('.')[0])
    return data,label

data,label = load_Img("./dolphins-and-seahorses/","dolphin")


import matplotlib.pyplot as plt
def plt_image(image):
    fig = plt.gcf()
    fig.set_size_inches(10,10)
    plt.imshow(image,cmap='binary')
    plt.show()

# plt_image(data[0])
# import cv2
# imgdir = os.listdir("./dolphins-and-seahorses/dolphin")
# print(imgdir)
# imageNum = len(imgdir)
# label = np.empty((imageNum,),dtype="uint8")
# data = []
# for i in range(imageNum):
#     data.append(cv2.imread("./dolphins-and-seahorses/dolphin/" + imgdir[i]))
#     label[i] = int(imgdir[i].split('.')[0])
#
# #labels = np.array(label)
# data = np.ndarray(data)
# print(len(data))
# print(label)
# print(data[2].shape)
# print(data)

import keras
#x_train4D = data.reshape(imageNum,179,300,3).astype('float32')
#print(x_train4D)
import cv2
def load_data(imgDir,imgFoldName):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = np.empty((imgNum,179,300,3),dtype="float32")
    label = np.empty((imgNum,),dtype="uint8")
    for i in range (imgNum):
        img = cv2.imread(imgDir+imgFoldName+"/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data = arr
        label[i] = int(imgs[i].split('.')[0])
    return data,label

data,label = load_data("./dolphins-and-seahorses/","dolphin")
print(data.shape)
print(label)

