#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
# @Time : 2019/9/7 13:37 
# @Author : zqy 
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 1、利用数据，在训练的时候实时提供数据
# mnist手写数字数据在运行时候实时提供给给占位符

tf.app.flags.DEFINE_integer("is_train", 0, "指定是否是训练模型，还是拿数据去预测")
FLAGS = tf.app.flags.FLAGS


# 构建多一层卷积网络
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def full_connected_mnist():
    """
    单层全连接神经网络识别手写数字图片
    特征值：[None, 784]
    目标值：one_hot编码 [None, 10]
    :return:
    """
    # 1、准备数据
    mnist = input_data.read_data_sets("./Mnist_data/", one_hot=True)
    # x [None, 784] y_true [None. 10]
    with tf.variable_scope("mnist_data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

    # 2、全连接层神经网络计算
    # 类别：10个类别  全连接层：10个神经元
    # 参数w: [784, 10]   b:[10]
    # 全连接层神经网络的计算公式：[None, 784] * [784, 10] + [10] = [None, 10]
    # 随机初始化权重偏置参数，这些是优化的参数，必须使用变量op去定义
    # with tf.variable_scope("fc_model"):
    #     weight = tf.Variable(tf.random_normal([784, 10]), name="w")
    #     bias = tf.Variable(tf.random_normal([10]), name="b")
    #     # fc层的计算
    #     # y_predict [None, 10]输出结果，提供给softmax使用
    #     y_predict = tf.matmul(x, weight) + bias

    # 第一层卷积
    with tf.variable_scope("conv1"):
        """
        卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，
        前两个维度是patch的大小，接着是输入的通道数目，
        最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。
        """
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        # 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
        # (因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        #我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    with tf.variable_scope("conv2"):
        """
        为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
        """
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    # 密集连接层
    with tf.variable_scope("intensive_con"):
        """
        现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
        我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
        """
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    with tf.variable_scope("Dropout"):
        """
        为了减少过拟合，我们在输出层之前加入dropout。
        我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
        这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
        TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
        所以用dropout的时候可以不用考虑scale。
        """
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.layers.dropout(h_fc1, keep_prob)

    with tf.variable_scope("softmax"):
        """最后，我们添加一个softmax层，就像前面的单层softmax regression一样。"""
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 3、softmax回归以及交叉熵损失计算
    with tf.variable_scope("softmax_crossentropy"):
        # labels:真实值 [None, 10]  one_hot
        # logits:全脸层的输出[None,10]
        # 返回每个样本的损失组成的列表
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
        loss = -tf.reduce_sum(y_true*tf.log(y_conv))

    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        # train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
    with tf.variable_scope("accuracy"):
        # equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # （2）收集要显示的变量
    # 先收集损失和准确率
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)
    # 收集权重和偏置
    tf.summary.histogram("weightes_fc1", W_fc1)
    tf.summary.histogram("weightes_fc2", W_fc2)
    tf.summary.histogram("biaes_fc1", b_fc1)
    tf.summary.histogram("biaes_fc2", b_fc2)

    # 初始化变量op
    # init_op = tf.global_variables_initializer()
    init_op = tf.initialize_all_variables()

    # （3）合并所有变量op
    merged = tf.summary.merge_all()

    # 创建模型保存和加载
    saver = tf.train.Saver()

    # 开启会话去训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # （1）创建一个events文件实例
        file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)

        # 加载模型
        if os.path.exists("./tmp/modelckpt/checkpoint"):
            saver.restore(sess, "./tmp/modelckpt/fc_nn_model")

        if FLAGS.is_train == 1:
            # 循环步数去训练

            for i in range(2000):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1.0})
                    train_loss = loss.eval(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1.0})
                    print("训练第%d步的准确率为: %g, 损失值为: %f" % (i, train_accuracy,train_loss))
                #训练
                train_op.run(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})
                # 运行合变量op，写入事件文件当中
                feed_dict = {x: batch[0], y_true: batch[1]}
                summary = sess.run(merged, feed_dict=feed_dict)
                file_writer.add_summary(summary, i)
                if i % 1000 == 0:
                    saver.save(sess, "./tmp/modelckpt/fc_nn_model")

        else:
            # # 如果不是训练，我们就去进行预测测试集数据
            for i in range(100):
                # 每次拿一个样本预测
                batch = mnist.train.next_batch(50)
                print("第%d个样本的真实值为：%d, 模型预测结果为：%d" % (
                    i + 1,
                    tf.argmax(sess.run(y_true, feed_dict={x: batch[0], y_true: batch[1]}), 1).eval(),
                    tf.argmax(sess.run(y_conv, feed_dict={x: batch[0], y_true: batch[1]}), 1).eval()
                )
                      )

        print("准确率为：", sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))

    return None


if __name__ == "__main__":
    full_connected_mnist()
