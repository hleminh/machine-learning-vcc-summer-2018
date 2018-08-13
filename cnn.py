from __future__ import print_function
from mnist import MNIST
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# Hyper parameter cua mo hinh

learning_rate = 0.001
momentum = 0.001
n_step = 400
mini_batch_size = 128

# Kien truc cua network

n_hidden_1 = 1024  # So luong unit trong hidden layer thu nhat
n_feature_map_1 = 32  # So luong feature map / unit trong convolutional layer thu nhat
n_feature_map_2 = 64  # So luong feature map / unit trong convolutional layer thu hai
n_input = 784  # So luong input feature(kich co anh 28*28 vay co 784 pixel)
n_class = 10  # So luong lop phan loai (chu so tu 0-9)
kernel_size = 5  # Kich thuoc kernel cua convolutional layer

# Du lieu input & output

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_class])

# Khai bao cac tham so (parameter)

w = {
    "c1": tf.Variable(tf.random_normal([kernel_size, kernel_size, 1, n_feature_map_1])),
    "c2": tf.Variable(tf.random_normal([kernel_size, kernel_size, n_feature_map_1, n_feature_map_2])),
    "h1": tf.Variable(tf.random_normal([7*7*n_feature_map_2, n_hidden_1])),
    "o": tf.Variable(tf.random_normal([n_hidden_1, n_class]))
}

b = {
    "c1": tf.Variable(tf.random_normal([32])),
    "c2": tf.Variable(tf.random_normal([64])),
    "h1": tf.Variable(tf.random_normal([n_hidden_1])),
    "o": tf.Variable(tf.random_normal([n_class]))
}

# De tien cho viec khai bao do thi tinh toan sau nay, tao ham tra lai 1 convolutional layer voi stride = 1 di kem voi relu layer


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Tuong tu, tao ham tra lai 1 maxpool layer voi kernel 2x2 va stride = 1


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Khai bao do thi tinh toan dau ra


def init_graph(x):
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


graph_output = init_graph(X)

# Khai bao ham mat mat va ham toi uu ham mat mat

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=graph_output, labels=Y))
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

# Khai bao ham danh gia mo hinh

pred = tf.nn.softmax(graph_output)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
score = tf.reduce_mean(tf.cast(correct_pred, "float"))

# Khai bao init cac variable
init = tf.global_variables_initializer()

# Toi uu ham mat
with tf.Session() as sess:
    sess.run(init)
    print ("Hoan thanh init!")
    print ("Bat dau toi uu...")
    for step in range(n_step+1):
        batch_x, batch_y = mnist.train.next_batch(mini_batch_size)
        sess.run([train_op], feed_dict={
            X: batch_x, Y: batch_y})
        if (step % 10 == 0) or (step == n_step):
            optimized_cost = cost.eval({X: batch_x, Y: batch_y})
            acc = score.eval(
                {X: batch_x, Y: batch_y})
            print("Step:", '%04d' %
                  (step), "cost={:.9f}".format(optimized_cost), "acc=", acc)

    print ("Da toi uu xong ham mat mat!")

    acc = score.eval({X: mnist.test.images[:256], Y: mnist.test.labels[:256]})
    print ("Do chinh xac: ", acc)
