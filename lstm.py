from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# Hyper parameter cua mo hinh
learning_rate = 0.001
n_training_step = 400
batch_size = 128

# Kien truc cua network
n_input = 28
n_time_step = 28
n_hidden = 128 # So luong unit trong hidden layer o moi gate cua LSTM
n_classes = 10  # So luong lop phan loai (chu so tu 0-9)

# Du lieu input & output
X = tf.placeholder("float", [None, n_time_step, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Khai bao cac tham so (parameter)

w = {
    'o': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
b = {
    'o': tf.Variable(tf.random_normal([n_classes]))
}


# Khai bao do thi tinh toan dau ra
def init_graph(x):

    x = tf.unstack(x, n_time_step, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], w['o']) + b['o']


graph_output = init_graph(X)
prediction = tf.nn.softmax(graph_output)

# Khai bao do thi tinh toan mat mat va ham toi uu ham mat mat

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=graph_output, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

# Khai bao do thi danh gia mo hinh

pred = tf.nn.softmax(graph_output)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
score = tf.reduce_mean(tf.cast(correct_pred, "float"))

# Khai bao init cac variable
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    print ("Hoan thanh init!")
    print ("Bat dau toi uu...")
    for step in range(1, n_training_step + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_time_step, n_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if (step % 10 == 0) or (step == n_training_step):
            optimized_cost, acc = sess.run([cost, score], feed_dict={X: batch_x,
                                                              Y: batch_y})
            print("Step: " + str(step) + ", cost= " +
                  "{:.4f}".format(optimized_cost) + ", acc= " +
                  "{:.3f}".format(acc))

    print ("Da toi uu xong ham mat mat!")

    test_data = mnist.test.images[:256].reshape(
        (-1, n_time_step, n_input))
    test_label = mnist.test.labels[:256]
    print("Do chinh xac:",
          sess.run(score, feed_dict={X: test_data, Y: test_label}))
