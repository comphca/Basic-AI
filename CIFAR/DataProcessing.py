import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import np_utils

#
(x_train,y_train),(x_test,y_test) = cifar10.load_data()


#x_train: 50000
#x_test: 10000
print('x_train:',len(x_train))
print('x_test:',len(x_test))

#(50000, 32, 32, 3)
#(50000, 1)
print(x_train.shape)
print(y_train.shape)

#
#print(x_train[0])

label_dict = {0:"airplan",1:"automobile",2:"bird",3:"cat",4:"deer",
              5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}