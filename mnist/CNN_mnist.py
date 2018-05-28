from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils


from util import show_train_history
from util import plot_image_labels_prediction

'''
	数据预处理
'''
#path='/home/comphca/anaconda3/envs/tensorflow/mnist.npz'
path = '../mnist.npz'
f = np.load(path)
x_Train, y_Train = f['x_train'], f['y_train']
x_Test, y_Test = f['x_test'], f['y_test']

#将图像特征值以reshape转换为60000*28*28*1的4维矩阵
#对比多层感知器，因为卷积网络要保证图像的维数，所以reshape转换为（个数×28×28×1）宽高和单色图像
x_Train4D = x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')
#print(x_Train.shape[0])
#print(x_Test.shape[0])
#print(x_Train4D[0])

#image图像标准化，除以255
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

#label处理，以One-Hot Encoding编码处理    0000000100则表示数字7
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_testOneHot = np_utils.to_categorical(y_Test)


'''
	建立模型
'''
#建立一个Sequential模型，后续使用model.add()增加网络层
model = Sequential()


#建立卷积层1，输入的数字图像大小为28*28，进行1次卷积运算产生16个图像，卷积不改变图像大小，所以还是28*28
#filters=16,建立16个滤镜  
#kernel_size=(5,5),每个滤镜5*5大小  
#padding='same',卷积运算产生的图像不改变大小     
#input_shape=(28,28,1),第一、二维表示输入的图像形状为28*28，第三维，因为是单色灰度图像，为1
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))

#建立池化层1
#输入参数pool_size=(2,2)，执行第一次缩减采样，将16个28*28的图像缩小为16个14*14的图像
model.add(MaxPooling2D(pool_size=(2,2)))


#建立卷积层2
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))

#建立池化层2
model.add(MaxPooling2D(pool_size=(2,2)))
#加入Dropout避免过拟合，随机放弃25%的神经元
model.add(Dropout(0.25))

#建立平坦层，共有36个7*7的图像转换为一维向量，长度是36*7*7=1764，也就是1764个float数，对应1764个神经元
model.add(Flatten())


#建立隐藏层，共有128个神经元
model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

#建立输出层，共有10个神经元，对应0-9数字，使用softmax激活函数进行转换，softmax可以将神经元的输出转换为预测每一个数字的概率
model.add(Dense(10,activation='softmax'))

#查看模型摘要
print(model.summary())


'''
	训练模型
'''
#loss：设置损失函数，optimizer：设置优化器，metrics：评估方式是准确率
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

#训练过程存在train_history 变量中
#validation_split表示keras训练时自动将数据划分为80%训练，20%验证，总共60000
#epochs=10表示执行10个训练周期batch_size=300每批300项数据，verbose=2设置显示训练过程
'''
	这里训练数据为48000个，48000/300=160，应该是160批次训练，所以是epochs=10的10个周期里面每次都训练160个组还是  10个周期是随机从160个周期里面选择
'''
train_history = model.fit(x = x_Train4D_normalize,
                          y = y_TrainOneHot,validation_split=0.2,
                          epochs=20,batch_size=300,verbose=2)


show_train_history.show_train_history(train_history,'acc','val_acc')

scores = model.evaluate(x_Test4D_normalize,y_testOneHot)
scores[1]


#预测
prediction = model.predict_classes(x_Test4D_normalize)
#查看钱10项数据
prediction[:10]



plot_image_labels_prediction(x_Test,y_Test,prediction,idx=0)