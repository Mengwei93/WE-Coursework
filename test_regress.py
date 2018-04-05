import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import random
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from data_process import *

####################### TENSORFLOW #######################

def main():
    train0 = pd.read_csv('/Users/mengwei.zhang/Desktop/train1_click.csv')
    test0 = pd.read_csv('/Users/mengwei.zhang/Desktop/test1_click.csv')
    val0 = pd.read_csv('/Users/mengwei.zhang/Desktop/val1_click.csv')

    train0 = pd.read_csv('/Users/mengwei.zhang/Desktop/train1_click.csv')
    test0 = pd.read_csv('/Users/mengwei.zhang/Desktop/test1_click.csv')
    val0 = pd.read_csv('/Users/mengwei.zhang/Desktop/val1_click.csv')


    train_bidprice = train0.bidprice
    train_payprice = train0.payprice
    train = train0.drop(['bidprice', 'payprice'], axis = 1)

    val_bidprice = val0.bidprice
    val_payprice = val0.payprice
    val = val0.drop(['bidprice', 'payprice'], axis = 1)

    test = test0
    test_norm = (test - test.mean()) / (test.max() - test.min())

    with tf.Session() as sess:
        data_ph = tf.placeholder(tf.float32, [None, 3])
        label_ph = tf.placeholder(tf.int32, [None])
        label = tf.one_hot(label_ph, 2)
        fc1 = tf.contrib.layers.fully_connected(data_ph, 500, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        fc2 = tf.contrib.layers.fully_connected(fc1, 500, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        logits = tf.contrib.layers.fully_connected(fc2, 2, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        predictions = tf.nn.softmax(logits)
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(label, 1)),tf.float32))

        tf.global_variables_initializer().run()

        dict = np.load('./model/price_step_100_0929PM_April_04_2018.npy').item()
        for var in tf.global_variables():
            dict_name = var.name
            if dict_name in dict:
                print('loaded var: %s' % dict_name)
                sess.run(var.assign(dict[dict_name]))
            else:
                print('missing var %s, skipped' % dict_name)

        # print(sess.run(accuracy, feed_dict={data_ph: data_test, label_ph: label_test}))
        final_price = sess.run(predictions, feed_dict={data_ph: test})
        # train_predict_price = sess.run(predictions, feed_dict={data_ph: train})
        # val_predict_price = sess.run(predictions, feed_dict={data_ph: val})
    print(final_price.shape)
    print(test.shape)

    test_price = pd.DataFrame(final_price)
    # test_price = pd.concat([val_predict_click,val_slotprice,val_payprice,val_bidprice],axis=1)
    test_price.to_csv('/Users/mengwei.zhang/Desktop/test_price.csv', index = False)

    # val_predict_click = pd.DataFrame(val_predict_click)
    # val1 = pd.concat([val_predict_click,val_slotprice,val_payprice,val_bidprice],axis=1)
    # val1.to_csv('/Users/mengwei.zhang/Desktop/val1_click.csv', index = False)

if __name__ == "__main__":
    main()
