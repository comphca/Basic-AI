import numpy as np
import os
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.utils import np_utils

all_img = []
path = "./dolphins-and-seahorses/train/"

all_img_list = os.listdir(path)
print(all_img_list)

all_img_num = len(all_img_list)  #6
print(all_img_list[0])

data = np.ones(shape=(all_img_num,400,400,3))
label = np.ones(shape=(all_img_num,),dtype=int)

#转换图片大小
def resize_img():
    for i in range(all_img_num):
        img = cv2.imread(path + all_img_list[i])
        img = cv2.resize(img,(400,400),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./out_dir/' + all_img_list[i],img)

resize_img()

for i in range(all_img_num):
    img = cv2.imread('./out_dir/' + all_img_list[i])
    data[i] = img
    label[i] = int(all_img_list[i].split('.')[0])

print(data.shape[0])
print(label)




train4D = data.reshape(data.shape[0],400,400,3).astype('float32')

train4D_noramlize = train4D / 255

y_trainOneHot = np_utils.to_categorical(label)

model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(400,400,3),
                 activation='relu'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,
                 kernel_size=(8,8),
                 padding='same',
                 activation='relu'
                 ))

model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(5,5)))

model.add(Conv2D(
    filters=64,
    kernel_size=(16,16),
    padding='same',
    activation='relu'
))
model.add(Dropout(rate=0.5))
model.add(MaxPooling2D(pool_size=(10,10)))
model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(2048,activation='relu'))

model.add(Dense(2,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])



#保存模型
try:
    model.load_weights("SaveModel/classic3.h5")
except:
    print("模型加载失败，开始训练")
    train_history = model.fit(x=train4D_noramlize,
                              y=y_trainOneHot,
                              validation_split=0.2,
                              epochs=30,
                              batch_size=5)
    model.save_weights("SaveModel/classic3.h5")


predict_data = np.ones(shape=(1,400,400,3))
preimg = cv2.imread("./predict_img/seahorse1.jpg")
preimg = cv2.resize(preimg,(400,400),interpolation=cv2.INTER_CUBIC)

predict_data[0] = img
pre4D = predict_data.reshape(predict_data.shape[0],400,400,3).astype('float32')
pre4D_nor = pre4D / 255

prediction = model.predict_classes(pre4D_nor)
print(prediction[0])


