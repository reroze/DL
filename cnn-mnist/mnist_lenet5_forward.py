#coding:utf-8
#搭建前行传播参数网络
import tensorflow as tf
IMAGE_SIZE = 28#分辨率28*28
NUM_CHANNELS = 1#通道数为1
CONV1_SIZE = 5#第一层卷积核大小为5
CONV2_SIZE = 5
CONV1_KERNEL_NUM = 32#第一层卷积核个数
CONV2_KERNEL_NUM = 64#第二层卷积核个数
FC_SIZE = 512#第一层有512个神经元
OUTPUT_NODE = 10#第二层有10个神经元（对应10分类输出）

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))#正则化
    #随机正则化初始一个shape形状的参数变量张量
    if regularizer != None: tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w#随机初始化


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))#全0
    return b


def conv2d(x, w):#x为4阶张量
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')#全0池化
#x为四阶张量 分别对应着数据集（batch），行分辨率（28），列分辨率（28），层数（1）
#w为一个卷积核的描述
#strides为卷积核移动的步长
#padding是池化方式（0填充）

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#最大池化
#x也是一个四阶张量


def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    #conv1_w是一个5行5列单通道32个形状的列表 即5*5*1*32
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    #偏置项是一个单通道的偏置项（应该与每一步卷积之后形成的单一个体相同的维度 此处由于conv1的个数为32个 因此最后得到的是一个32通道的卷积结果，因此平行的偏置项也为32维）
    conv1 = conv2d(x, conv1_w)
    #构造卷积，并返回卷积结果
    #conv1应该是一个28*28*32
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))#bias_add为添加偏置
    #激活函数
    pool1 = max_pool_2x2(relu1)
    #pool1应该是一个14*14*32的一个张量
#第一次池化操作

#将第一次池化输出的结果pool1作为参数数据进行下一次的训练数据（即与原来x的地位相同）
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    #conv2_w应该是一个5*5*32*64的一个张量
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    #relu2应该是一个14*14*64
    pool2 = max_pool_2x2(relu2)
    #pool2应该是一个7*7*64
    #pool2应该是7*7*64维
    #shape[0]固定为1？
#第二次池化操作
    #需要将pool2从三维张量变为二维张量
    pool_shape = pool2.get_shape().as_list()
    #shape【0】是一个batch的值，【1】，【2】，【3】分别代表提取特征的长度宽度和深度
    #shape中的每一个元素分别表示（pool2中每一个维数的大小）
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]#将多维张量一维化变为参数输入到神经网络进行训练
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])#二维化为batch数*特征点个数（喂入多少个样本对应多少行）（每一行都对应着nodes个特征点的个数）
    #将此作为新的数据集【矩阵】进行全连接网络分析 训练参数

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
#获取随机初始化的参数的形式是一样的（都是传入对应的shape 进行正则化随机就行）
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    #fc1是【batch， nodes】*【nodes， FC_SIZE】=【batch， FC_SIZE】
    #训练单张的话应该是【1， FC_SIZE】
    if train: fc1 = tf.nn.dropout(fc1, 0.5)
    #如果是训练阶段 启用dropout函数减少过拟合现象
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    #fc2_w是【batch，FC_SIZE】*【FC_SIZE， 10】=【batch， 10】
    #测试为1张时，输出为一个【1*10】的列表，表中数值最大的即为预测数字
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
#全连接神经网络
