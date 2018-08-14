from __future__ import print_function
from mnist import MNIST
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# -------------- Khai bao hyper parameter --------------

learning_rate = 0.001
n_step = 400
mini_batch_size = 128

# -------------- Khai bao kien truc cua mo hinh --------------

n_hidden_1 = 1024  # So luong unit trong hidden layer thu nhat
n_hidden_2 = 1024  # So luong unit trong hidden layer thu hai
n_input = 784  # So luong input feature(kich co anh 28*28 vay co 784 pixel)
n_class = 10  # So luong lop phan loai (chu so tu 0-9)

# -------------- Khai bao placeholder chua data & label --------------

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_class])

# -------------- Khai bao tham so cua mo hinh --------------

w = {
    "h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    "h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    "o": tf.Variable(tf.random_normal([n_hidden_2, n_class]))
}

b = {
    "h1": tf.Variable(tf.random_normal([n_hidden_1])),
    "h2": tf.Variable(tf.random_normal([n_hidden_2])),
    "o": tf.Variable(tf.random_normal([n_class]))
}

# -------------- Khai bao cac operation --------------

# Khai bao operation tinh toan dau ra


def mlp(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w["h1"]), b["h1"]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w["h2"]), b["h2"]))
    layer_output = tf.add(tf.matmul(layer_2, w["o"]), b["o"])
    return layer_output


nn_output = mlp(X)
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

    test_acc = acc.eval({X: mnist.test.images, Y: mnist.test.labels})
    print ("Do chinh xac: ", test_acc)
