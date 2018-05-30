import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import np_utils


from util.CIFAR.plot_image_label_prediction import plot_image_labels_prediction

#加载数据
(x_train,y_train),(x_test,y_test) = cifar10.load_data()


#x_train: 50000
#x_test: 10000
#数据项数
print('x_train:',len(x_train))
print('x_test:',len(x_test))

#(50000, 32, 32, 3)
#(50000, 1)  50000项数据，50000*1的矩阵，每个一维向量表示一种类别
#图像shape形状，第一维是项数，二三维是图像大小32*32，第四维是RGB三原色所以是3
print(x_train.shape)
print(y_train.shape)

#表示第一个图像，是32*32*3的矩阵表示
#print(x_train[0])

#建数据字典，定义每个数字代表的图像类别
label_dict = {0:"airplan",1:"automobile",2:"bird",3:"cat",4:"deer",
              5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

#
#plot_image_labels_prediction(x_train,y_train,[],0)



#[59 62 63]
print(x_train[0][0][0])

#!!!!
x_train_normalize = x_train.reshape(x_train.shape[0],32,32,3).astype('float32') / 255
x_test_normalize = x_test.reshape(x_test.shape[0],32,32,3).astype('float32') / 255

#[0.23137255 0.24313726 0.24705882]
print(x_train_normalize[0][0][0])



#label
#(50000, 1)
print(y_train.shape)

#[[6][9][9][4][1]]
print(y_train[:5])

y_train_OneHot = np_utils.to_categorical(y_train)
y_test_OneHot = np_utils.to_categorical(y_test)

print(y_train_OneHot.shape)
print(y_train_OneHot[:5].astype('int'))