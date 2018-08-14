from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# -------------- Khai bao hyper parameter --------------

learning_rate = 0.001
n_training_step = 400
batch_size = 128

# -------------- Khai bao kien truc cua mo hinh --------------

n_input = 28
n_time_step = 28
n_hidden = 256  # So luong unit trong hidden layer o moi gate cua LSTM
n_classes = 10  # So luong lop phan loai (chu so tu 0-9)

# -------------- Khai bao placeholder chua data & label --------------

X = tf.placeholder("float", [None, n_time_step, n_input])
Y = tf.placeholder("float", [None, n_classes])

# -------------- Khai bao tham so cua mo hinh --------------

w = {
    'o': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
b = {
    'o': tf.Variable(tf.random_normal([n_classes]))
}

# -------------- Khai bao cac operation --------------

# Khai bao operation tinh toan dau ra


def lstm(x):

    x = tf.unstack(x, n_time_step, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], w['o']) + b['o']


nn_output = lstm(X)
pred = tf.nn.softmax(nn_output)

# Khai bao operation tinh ham mat mat va operation toi uu ham mat mat

cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=nn_output, labels=Y))
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
    for step in range(1, n_training_step + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_time_step, n_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if (step % 10 == 0) or (step == n_training_step):
            optimized_cost, training_acc = sess.run(
                [cost_op, acc], feed_dict={X: batch_x, Y: batch_y})
            print("Step: " + str(step) + ", cost= " +
                  "{:.4f}".format(optimized_cost) + ", acc= " + "{:.3f}".format(training_acc))

    print ("Da toi uu xong ham mat mat!")

# Danh gia do chinh xac cua mo hinh

    test_data = mnist.test.images[:256].reshape(
        (-1, n_time_step, n_input))
    test_label = mnist.test.labels[:256]
    test_acc = sess.run(acc, feed_dict={X: test_data, Y: test_label})
    print("Do chinh xac:", test_acc)
