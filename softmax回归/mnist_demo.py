#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
# @Time : 2019/9/7 12:05 
# @Author : zqy 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 准备数据
mnist = input_data.read_data_sets("./Mnist_data/", one_hot=True)

# 可交互的操作单元'占位符',希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量(28*28)
x = tf.placeholder("float", [None, 784])  # None表示此张量的第一个维度可以是任何长度的

# 权重值和偏置量
# 一个Variable代表一个可修改的张量
W = tf.Variable(tf.zeros([784, 10]))  # 我们想要用784维的图片向量乘以它以得到一个10维的证据值向量
b = tf.Variable(tf.zeros([10]))  # 形状是[10]

# 实现模型
# tf.matmul(​​X，W)表示x乘以W
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float", [None, 10])

# 计算交叉熵
# tf.reduce_sum 计算张量的所有元素的总和
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # tf.log 计算 y 的每个元素的对数

# 以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 添加一个操作来初始化我们创建的变量：
init = tf.initialize_all_variables()

# 我们可以在一个Session里面启动我们的模型，并且初始化变量：
sess = tf.Session()
sess.run(init)

# 然后开始训练模型，这里我们让模型循环训练1000次！
for i in range(1000):
    # 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，
    # 然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

