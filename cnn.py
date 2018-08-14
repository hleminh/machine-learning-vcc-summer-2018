from __future__ import print_function
from mnist import MNIST
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# -------------- Khai bao hyper parameter --------------

learning_rate = 0.001
n_step = 400
mini_batch_size = 128

# -------------- Khai bao kien truc cua mo hinh --------------

n_hidden = 1024  # So luong unit trong hidden layer
n_feature_map_1 = 32  # So luong feature map / unit trong convolutional layer thu nhat
n_feature_map_2 = 64  # So luong feature map / unit trong convolutional layer thu hai
n_input = 784  # So luong input feature(kich co anh 28*28 vay co 784 pixel)
n_class = 10  # So luong lop phan loai (chu so tu 0-9)
kernel_size = 5  # Kich thuoc kernel cua convolutional layer

# -------------- Khai bao placeholder chua data & label --------------

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_class])

# -------------- Khai bao tham so cua mo hinh --------------

w = {
    "c1": tf.Variable(tf.random_normal([kernel_size, kernel_size, 1, n_feature_map_1])),
    "c2": tf.Variable(tf.random_normal([kernel_size, kernel_size, n_feature_map_1, n_feature_map_2])),
    "h1": tf.Variable(tf.random_normal([7*7*n_feature_map_2, n_hidden])),
    "o": tf.Variable(tf.random_normal([n_hidden, n_class]))
}

b = {
    "c1": tf.Variable(tf.random_normal([32])),
    "c2": tf.Variable(tf.random_normal([64])),
    "h1": tf.Variable(tf.random_normal([n_hidden])),
    "o": tf.Variable(tf.random_normal([n_class]))
}

# -------------- Khai bao cac operation --------------

# Khai bao operation tao convolutional layer


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Khai bao operation tao maxpool layer


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Khai bao operation tinh toan dau ra


def cnn(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv_1 = conv2d(x, w["c1"], b["c1"])
    conv_1 = maxpool2d(conv_1)
    conv_2 = conv2d(conv_1, w["c2"], b["c2"])
    conv_2 = maxpool2d(conv_2)
    fc_1 = tf.reshape(conv_2, [-1, w["h1"].get_shape().as_list()[0]])
    fc_1 = tf.add(tf.matmul(fc_1, w["h1"]), b["h1"])
    fc_1 = tf.nn.relu(fc_1)
    output_layer = tf.add(tf.matmul(fc_1, w["o"]), b["o"])
    return output_layer


nn_output = cnn(X)
pred = tf.nn.softmax(nn_output)

# Khai bao operation tinh ham mat mat va operation toi uu ham mat mat

cost_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=nn_output, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost_op)

# Khai bao operation danh gia mo hinh

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, "float"))

# Khai bao operation initialize cac variable

init = tf.global_variables_initializer()

# -------------- Huan luyen neural network & danh gia mo hinh --------------

# Huan luyen network

with tf.Session() as sess:

    sess.run(init)
    print ("Hoan thanh init!")
    print ("Bat dau toi uu...")
    for step in range(n_step+1):
        batch_x, batch_y = mnist.train.next_batch(mini_batch_size)
        sess.run([train_op], feed_dict={X: batch_x, Y: batch_y})
        if (step % 10 == 0) or (step == n_step):
            optimized_cost, training_acc = sess.run(
                [cost_op, acc], feed_dict={X: batch_x, Y: batch_y})
            print("Step:", '%04d' % (step), "cost={:.9f}".format(
                optimized_cost), "acc=", training_acc)

    print ("Da toi uu xong ham mat mat!")

# Danh gia do chinh xac cua mo hinh

    test_acc = acc.eval(
        {X: mnist.test.images[:256], Y: mnist.test.labels[:256]})
    print ("Do chinh xac: ", test_acc)
