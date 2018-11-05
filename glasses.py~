
import tensorflow as tf
import numpy as np
import random as rand
import array as ar
import os
from sklearn.model_selection import train_test_split

DATA_DIR = './Dataset'
NC = 2
STEPS = 10000
LR = 0.0001

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def getData(data='conv'):

    imgf = DATA_DIR+'/'+data+'-images-idx3-ubyte'
    lblf = DATA_DIR+'/'+data+'-labels-idx1-ubyte'

    structh = np.dtype([('magic','>i4'),('N','>i4'),('r','>i4'),('c','>i4')])

    fd = open(imgf, 'rb')

    header = np.fromfile(fd, dtype=structh, count=1)
    N = header['N'][0]
    r = header['r'][0]
    c = header['c'][0]
    im = ar.array('B', fd.read())

    fd.close()

    fd = open(lblf, 'rb')

    structh = np.dtype([('magic','>i4'),('N','>i4')])
    header = np.fromfile(fd, dtype=structh, count=1)
    lb = ar.array('B', fd.read())
    
    images = []
    labels = []

    for i in range(N):
        images.append(im[i*r*c:(i+1)*r*c])
        l = [0]*NC
        l[lb[i]] = 1
        labels.append(l)
    
    images = np.array(images).reshape((N,r*c))
    labels = np.array(labels).reshape((N,NC))

    return images, labels, r, c

if __name__ == '__main__':

    x_train, y_train, r, c = getData(data='all')
    x_test, y_test, r, c = getData(data='all')

    N_train = len(x_train)
    size_train = len(x_train[0])
    
    N_test = len(x_test)
    size_test = len(x_test[0])

    x = tf.placeholder(tf.float32, [None, size_train])
    w = tf.Variable(tf.zeros([size_train, NC]))

    y = tf.placeholder(tf.float32, [None, NC])

    y_pred = tf.matmul(x, w)
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y))
    optimizer_train = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(STEPS):
            if i%100 == 0:
                ans = sess.run(accuracy, feed_dict={x: x_test, y:y_test})
                print('Steps: {0:6d}, Accuracy: {1:.2f}'.format(i, ans*100))
            sess.run(optimizer_train, feed_dict={x: x_train, y: y_train})
            
        ans = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print(ans*100)
