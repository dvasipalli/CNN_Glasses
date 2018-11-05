
import tensorflow as tf
import numpy as np
import random as rand
import os
import sys
from glasses import getData
from sklearn.model_selection import train_test_split

STEPS = 50000
BATCH = 20
LR = 0.00001
NC = 2
NODES = 50

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)
def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b
def get_batch(x_train, y_train, N):
    l = rand.randint(0, N-BATCH)
    r = l + BATCH
    return x_train[l:r], y_train[l:r]

if __name__ == "__main__":

    try:
        convdim = int(sys.argv[1])
    except:
        print ('Usage: {} <int>'.format(sys.argv[0]))
        sys.exit()
    images, labels, r, c = getData(data='original')
    x_train, x_test, y_train, y_test = train_test_split(images, labels)
    
    N_train = len(x_train)
    size_train = len(x_train[0])
    
    N_test = len(x_test)
    size_test = len(x_test[0])

    x = tf.placeholder(tf.float32, [None, size_train])
    y = tf.placeholder(tf.float32, [None, NC])
    x_image = tf.reshape(x, [-1, r, c, 1])
    
    conv1 = conv_layer(x_image, shape=[convdim, convdim, 1, 32])
    # (?, 112, 92, 32)
    conv1_pool = max_pool_2x2(conv1)
    # (?, 56, 46, 32)
    
    conv2 = conv_layer(conv1_pool, shape=[convdim, convdim, 32, 64])
    # (?, 56, 46, 64)
    conv2_pool = max_pool_2x2(conv2)
    # (?, 28, 23, 64)
    
    conv3 = conv_layer(conv2_pool, shape=[convdim, convdim, 64, 128])
    # (?, 28, 23, 128)
    conv3_pool = max_pool_2x2(conv3)
    # (?, 14, 12, 128)
    
    conv4 = conv_layer(conv3_pool, shape=[convdim, convdim, 128, 256])
    # (?, 14, 12, 128)
    conv4_pool = max_pool_2x2(conv4)
    # (?, 7, 6, 256)

    conv_flat = tf.reshape(conv4_pool, [-1, 7*6*256])
    full1 = tf.nn.relu(full_layer(conv_flat, NODES))

    keep_prob = tf.placeholder(tf.float32)
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, NC)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y))
    train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
    correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            if i%100 == 0:
                ans = sess.run(accuracy, feed_dict={x:x_test,y:y_test, keep_prob:1})
                print('Steps: {0:6d}, Accuracy: {1:.2f}'.format(i, ans*100))

            x_batch, y_batch = get_batch(x_train, y_train, N_train)
            sess.run(train_step, feed_dict={x: x_batch, y: y_batch, keep_prob:0.7})

        ans = sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:1})
        print(ans*100)
