"""
神经网络模型相关
RNN-LSTM 循环神经网络

大家之后可以加上各种的 name_scope（命名空间）
用 TensorBoard 来可视化

==== 一些术语的概念 ====
# Batch size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目。Batch size 越大，所需的内存就越大
# Iteration : 迭代。每一次迭代更新一次权重（网络参数），每一次权重更新需要 Batch size 个数据进行 Forward 运算，再进行 BP 运算
# Epoch : 纪元/时代。所有的训练样本完成一次迭代

# 假如 : 训练集有 1000 个样本，Batch_size=10
# 那么 : 训练完整个样本集需要： 100 次 Iteration，1 个 Epoch
# 但一般我们都不止训练一个 Epoch

==== 超参数（Hyper parameter）====
init_scale : 权重参数（Weights）的初始取值跨度，一开始取小一些比较利于训练
learning_rate : 学习率，训练时初始为 1.0
num_layers : LSTM 层的数目（默认是 2）
num_steps : LSTM 展开的步（step）数，相当于每个批次输入单词的数目（默认是 35）
hidden_size : LSTM 层的神经元数目，也是词向量的维度（默认是 650）
max_lr_epoch : 用初始学习率训练的 Epoch 数目（默认是 10）
dropout : 在 Dropout 层的留存率（默认是 0.5）
lr_decay : 在过了 max_lr_epoch 之后每一个 Epoch 的学习率的衰减率，训练时初始为 0.93。让学习率逐渐衰减是提高训练效率的有效方法
batch_size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目
（batch_size 默认是 20。取比较小的 batch_size 更有利于 Stochastic Gradient Descent（随机梯度下降），防止被困在局部最小值）
"""
import tensorflow as tf

#神经网络模型
class Model(object):
    #构造函数
    def __init__(self,input_obj, is_training, hidden_size, vocab_size, num_layers, dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input_obj
        self.batch_size = input_obj.batch_size
        self.num_steps = input_obj.num_steps
        self.hidden_size = hidden_size

        # 让这里的操作和变量用 CPU 来计算，因为暂时（貌似）还没有 GPU 的实现
        with tf.device("/cpu:0"):
            # 创建 词向量（Word Embedding），Embedding 表示 Dense Vector（密集向量）
            # 词向量本质上是一种单词聚类（Clustering）的方法
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            # embedding_lookup 返回词向量
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)
