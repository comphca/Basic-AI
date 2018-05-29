import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import np_utils

#加载数据
(x_train,y_train),(x_test,y_test) = cifar10.load_data()


#x_train: 50000
#x_test: 10000
#数据项数
print('x_train:',len(x_train))
print('x_test:',len(x_test))

#(50000, 32, 32, 3)
#(50000, 1)  50000项数据，每个是0-9的表示，每个数字代表一个类别
#图像shape形状，第一维是项数，二三维是图像大小32*32，第四维是RGB三原色所以是3
print(x_train.shape)
print(y_train.shape)

#表示第一个图像，是32*32*3的矩阵表示
#print(x_train[0])

#建数据字典，定义每个数字代表的图像类别名称
label_dict = {0:"airplan",1:"automobile",2:"bird",3:"cat",4:"deer",
              5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}