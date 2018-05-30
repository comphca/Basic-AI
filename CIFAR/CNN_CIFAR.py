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
x_train_normalize = x_train.astype('float') / 255
x_test_normalize = x_test.astype('float') / 255
#print(x_train_normalize[0][0][0])

y_train_OneHot = np_utils.to_categorical(y_train)
y_test_OneHot = np_utils.to_categorical(y_test)

#model
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32,32,3),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(10,activation='softmax'))
#print(model.summary())


#train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

train_history = model.fit(x_train_normalize,y_train_OneHot,
                          validation_split=0.2,
                          epochs=10,batch_size=128,verbose=1)