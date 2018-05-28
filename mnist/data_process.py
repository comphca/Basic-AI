import numpy as np
import pandas as pd
from keras.utils import np_utils


'''
    函数
'''
import matplotlib.pyplot as plt

#显示数字图像
def plot_image(image):
    #设置显示图形大小
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    #使用imshow显示图形，传入图形是28×28大小，cmap参数设置为binary以黑白灰度显示
    plt.imshow(image,cmap='binary')
    plt.show()

#查看数字图形、真实数字和预测结果
#images （数字图像）,labels （真实值）,prediction （预测结果）,idx （开始显示的数据index）,num=10（要显示的个数，默认为10）
def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    #设置显示图形大小
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    #参数项如果大于25就为25
    if num > 25: num = 25
    #画出num个数字图像
    for i in range(0,num):
        #建立subgraph子图形为5行5列
        ax = plt.subplot(5,5,1+i)
        #画出subgraph子图形
        ax.imshow(images[idx],cmap='binary')
        #设置子图像title，显示标签字段
        title = "label =" + str(labels[idx])
        #如果传入了预测结果
        if len(prediction) > 0:
            #标题
            title += ",predict=" + str(prediction[idx])
        #设置子图像的标题和刻度
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        #读取下一项
        idx += 1
    plt.show()



#加载数据
path = '../mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

#train data =  60000
#test data =  10000
print('train data = ',len(x_train))
print('test data = ',len(x_test))


#x_train: (60000, 28, 28)
#y_train: (60000,)
#x与y共60000项数据，x是单色数字图像28*28大小，y是数字图像的真实值，shape是numpy里面的函数，读取矩阵长度
print('x_train:',x_train.shape)
print('y_train:',y_train.shape)
#print(x_train.shape[0],"   ",x_train.shape[1]*x_train.shape[2])



#plot_image(x_train[0])
#print(y_train[0])


#plot_image_labels_prediction(x_train,y_train,[],0,10)


#x_Train: (60000, 784)
#x_Test: (10000, 784)
#float为转换后数值表示形式
x_Train = x_train.reshape(60000,784).astype('float32')
x_Test = x_test.reshape(10000,784).astype('float32')
print('x_Train:',x_Train.shape)
print('x_Test:',x_Test.shape)

#28*28的矩阵
#print(x_train[0])
print("--------------------------------------------------------")
#1×784的向量(x_Train为60000×784的矩阵)
print(x_Train[0])
print("--------------------------------------------------------")



x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255


'''
    label数据处理
'''
#[5 0 4 1 9]
print(y_train[:5])

y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

#[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
print(y_TrainOneHot[:5])


