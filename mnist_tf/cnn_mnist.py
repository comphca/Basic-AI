import numpy as np
import tensorflow as tf

#下载mnist数据集（55000 * 28 * 28）
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data',one_hot=True)


#None 表示张量（Tensor）得第一个维度可以是任何长度 原本图像是28*28得像素
input_x = tf.placeholder(tf.float32,[None, 28 * 28]) / 255.
#输出是10个值
output_y = tf.placeholder(tf.int32, [None, 10])

input_x_images = tf.reshape(input_x,[-1,28,28,1])  #改变形状之后得输入

#从Test（测试）数据集里面选取3000个手写数字得图片和对应标签
test_x = mnist.test.images[:3000] #图片
test_y = mnist.test.labels[:3000] #标签

#构建卷积神经网络
# 第 1 层卷积
conv1 = tf.layers.conv2d(
    inputs=input_x_images, #形状[28*28*1]
    filters=32,            #过滤器数目 32个，输出深度是32
    kernel_size=(5,5),     #过滤器在二位得大小事（5*5）
    strides=1,             #步长是1
    padding='same',        #same表示输出的大小不变，需要在外围补零两圈
    activation=tf.nn.relu  #激活函数是Relu
)  #形状[ 28 , 28 , 32]

#第 1 层池化（亚采样）
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,   #形状[28 , 28, 32]
    pool_size=(2,2),#过滤器在二维里面得大小
    strides=2       #步长2
) #形状[14 , 14 , 32]


# 第 2 层卷积
conv2 = tf.layers.conv2d(
    inputs=pool1,          #形状[14*14*32]
    filters=64,            #过滤器数目 64个，输出深度是64
    kernel_size=(5,5),     #过滤器在二位得大小事（5*5）
    strides=1,             #步长是1
    padding='same',        #same表示输出的大小不变，需要在外围补零两圈
    activation=tf.nn.relu  #激活函数是Relu
)  #形状[ 14 , 14 , 64]

#第 2 层池化（亚采样）
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,   #形状[14 , 14, 64]
    pool_size=(2,2),#过滤器在二维里面得大小
    strides=2       #步长2
) #形状[7 , 7 , 64]

#平坦化（flat）  -1是根据之后得参数推断-1这个位置上维度得大小
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])   #[7 * 7 * 64]

#1024个神经元得全连接层
dense = tf.layers.dense(inputs=flat,units=1024,activation=tf.nn.relu)

#Dropout:丢弃50% rate = 0.5
dropout = tf.layers.dropout(inputs=dense,rate=0.5)

#10个神经元得全连接层，这里不用激活函数来做非线性化
logits = tf.layers.dense(inputs=dropout,units=10) #输出。形状[1, 1, 10]


#计算误差（计算Cross entropy（交叉熵）），再用Softmax计算百分比概率
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

#用优化器Adam最小化误差，学习率0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#精度  计算 预测值  和 实际标签  得匹配程度
#返回  （accuracy， update_op）, 会创建两个局部变量
accuracy = tf.metrics.accuracy(
    labels=tf.arg_max(output_y,dimension=1),
    predictions=tf.arg_max(logits,dimension=1),)[1]   #logits, axis = 1


#创建会话
sess = tf.Session()
#初始化变量:全局和局部
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50)   #从train（训练数据集）里取下一个 50 个样本
    train_loss, train_op_ = sess.run([loss, train_op], {input_x:batch[0],output_y:batch[1]})
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy,{input_x: test_x, output_y: test_y})
        print("Step=%d, Train loss=%.4f, [Test accuracy = %.2f]" %(i,train_loss,test_accuracy))


#测试打印20个预测值 和 真实值  得对
test_output = sess.run(logits, {input_x: test_x[:20]})
inferenced_y = np.argmax(test_output,1)
print(inferenced_y, 'Inferenced number') #推测得数字
print(np.argmax(test_y[:20], 1), 'Real number') #真实得数字