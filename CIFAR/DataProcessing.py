import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import np_utils

#��������
(x_train,y_train),(x_test,y_test) = cifar10.load_data()


#x_train: 50000
#x_test: 10000
#��������
print('x_train:',len(x_train))
print('x_test:',len(x_test))

#(50000, 32, 32, 3)
#(50000, 1)  50000�����ݣ�ÿ����0-9�ı�ʾ��ÿ�����ִ���һ�����
#ͼ��shape��״����һά������������ά��ͼ���С32*32������ά��RGB��ԭɫ������3
print(x_train.shape)
print(y_train.shape)

#��ʾ��һ��ͼ����32*32*3�ľ����ʾ
#print(x_train[0])

#�������ֵ䣬����ÿ�����ִ����ͼ���������
label_dict = {0:"airplan",1:"automobile",2:"bird",3:"cat",4:"deer",
              5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}