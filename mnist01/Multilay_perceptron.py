from keras.utils import np_utils
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

#加载数据
path = '../mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']


x_Train = x_train.reshape(60000,784).astype('float32')
x_Test = x_test.reshape(10000,784).astype('float32')

#image图像标准化，除以255
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)


model = Sequential()


#建立输入层和隐藏层，使用add方法加入Dense神经网络层，特点所有上一层和下一层神经元完全连接

    #units表示隐藏层神经元个数为256
    #input_dim表示输入层神经元个数为784（28×28=784）
    #kernel_initializer表示使用normal distribution正态分布的随机数来初始化权重和bias（偏差）
    #定义激活函数为relu
model.add(Dense(units=256,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))



#建立输出层

    #units  定义“输出层”神经元个数微10
    #kernel_initializer  使用normal distribution正态分布的随机数来初始化权重和bias（偏差）
    #activation  定义激活函数为softmax
    #建立Dense神经网络不需要设置input_dim，keras会根据上一层units256个设置这一层为256

model.add(Dense(units=10,
                 kernel_initializer='normal',
                 activation='softmax'))



'''
    训练模型
'''
#训练前定义训练方式
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

#训练模型
train_history = model.fit(x = x_Train_normalize,
                          y = y_TrainOneHot,validation_split=0.2,
                          epochs=10,batch_size=200,verbose=2)