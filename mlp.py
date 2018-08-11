from __future__ import print_function
from mnist import MNIST
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# Hyper parameter cua mo hinh

learning_rate = 0.001
n_epoch = 50
mini_batch_size = 10

# Kien truc cua network

n_hidden_1 = 256  # So luong unit trong hidden layer thu nhat
n_hidden_2 = 256  # So luong unit trong hidden layer thu hai
n_input = 784  # So luong input feature(kich co anh 28*28 vay co 784 pixel)
n_class = 10  # So luong lop phan loai (chu so tu 0-9)

# Du lieu input & output

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_class])

# Khai bao cac tham so (parameter)

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

# Khai bao do thi tinh toan dau ra


def init_graph(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w["h1"]), b["h1"]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w["h2"]), b["h2"]))
    layer_output = tf.add(tf.matmul(layer_2, w["o"]), b["o"])
    return layer_output


graph_output = init_graph(X)

# Khai bao ham mat mat va ham toi uu ham mat mat

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=graph_output, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost)

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
    for epoch in range(n_epoch + 1):
        total_cost = 0
        total_score = 0
        n_batch = int(mnist.train.num_examples/mini_batch_size)
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(mini_batch_size)
            _, optimized_cost, batch_score = sess.run([optimizer, cost, score], feed_dict={
                X: batch_x, Y: batch_y})
            total_cost += optimized_cost
            total_score += batch_score
        avg_cost = total_cost/n_batch
        avg_score = total_score/n_batch
        if (epoch % 10 == 0) or (epoch == n_epoch):

            print("Epoch: " + str(epoch) + ", cost= " +
                  "{:.4f}".format(avg_cost) + ", acc= " +
                  "{:.3f}".format(avg_score))

    print ("Da toi uu xong ham mat mat!")
    print ("cost = {:.9f}".format(avg_cost))

    acc = score.eval({X: mnist.test.images, Y: mnist.test.labels})
    print ("Do chinh xac: ", acc)
