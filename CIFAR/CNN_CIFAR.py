from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils


#加载数据
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#train_data: image: (50000, 32, 32, 3) labels: (50000, 1)
#test_data: image: (10000, 32, 32, 3) labels: (10000, 1)
print('train_data:','image:',x_train.shape,
      'labels:',y_train.shape)
print('test_data:','image:',x_test.shape,
      'labels:',y_test.shape)


#[0.23137255 0.24313725 0.24705882]   train[0][0][0]
#这里作用和 x_train.reshape(x_train.shape[0],32,32,3).astype('float') / 255 一样效果，为了后面卷积操作，维数不能降低
x_train_normalize = x_train.astype('float') / 255
x_test_normalize = x_test.astype('float') / 255
#print(x_train_normalize[0][0][0])

y_train_OneHot = np_utils.to_categorical(y_train)
y_test_OneHot = np_utils.to_categorical(y_test)

#model
model = Sequential()
#32个卷积核，3*3大小
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32,32,3),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.25))

#缩减每幅图为16*16大小
model.add(MaxPooling2D(pool_size=(2,2)))

#64个卷积核，3*3大小
model.add(Conv2D(filters=64,kernel_size=(3,3),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.25))

#缩减每幅图为8*8大小
model.add(MaxPooling2D(pool_size=(2,2)))

#建立平坦层，共有64个8*8的图像转换为一维向量，长度是64*8*8=4096，4096个float数，对应4096个神经单元
model.add(Flatten())
model.add(Dropout(rate=0.25))

#建立隐藏层，共1024个单元
model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.25))

#建立输出层，10个神经单元，使用softmax激活函数进行转换，softmax可以将神经元的输出转换为预测每一个图像类别的概率
model.add(Dense(10,activation='softmax'))
#print(model.summary())


#train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])


#validation_split=0.2  将train数据分为80%作为训练数据，20%作为验证数据 50000*80% = 40000
#epochs=10  执行10个训练周期  40000/128 = 312  所以每个训练周期大概分为312次训练
#batch_size=128   1/10 --> 128/40000   256/40000  384/40000
train_history = model.fit(x_train_normalize,y_train_OneHot,
                          validation_split=0.2,
                          epochs=10,batch_size=128,verbose=1)