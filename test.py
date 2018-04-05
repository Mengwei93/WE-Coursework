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
    train0 = pd.read_csv('/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/down_sample/train3.csv')
    test0 = pd.read_csv('/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/down_sample/test3.csv')
    val0 = pd.read_csv('/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/down_sample/validation3.csv')

    train_slotprice = train0.slotprice
    train_bidprice = train0.bidprice
    train_payprice = train0.payprice
    train_click = train0.click
    train = train0.drop(['slotprice', 'payprice', 'bidprice','click'], axis = 1)

    val0 = negative_down_sampling(val0, 0.25)
    val_slotprice = val0.slotprice
    val_bidprice = val0.bidprice
    val_payprice = val0.payprice
    val_click = val0.click
    val = val0.drop(['slotprice', 'payprice', 'bidprice','click'], axis = 1)


    test_slotprice = test0.slotprice
    test = test0.drop(['slotprice'], axis = 1)


    # to have same number of columns, train drop 5, test drop 10
    train = train.drop(['usertag_13874', 'usertag_14273', 'usertag_16617', 'usertag_16661', 'usertag_16753'], axis = 1)
    
    test = test.drop(['usertag_13874', 'usertag_14273', 'usertag_16617', 'usertag_16661', 'usertag_16753'], axis = 1)
    test = test.drop(['usertag_13496', 'usertag_13776', 'usertag_13800', 'usertag_13866', 'usertag_16706'], axis = 1)

    with tf.Session() as sess:
        data_ph = tf.placeholder(tf.float32, [None, 593])
        label_ph = tf.placeholder(tf.int32, [None])
        label = tf.one_hot(label_ph, 2)
        fc1 = tf.contrib.layers.fully_connected(data_ph, 500, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        fc2 = tf.contrib.layers.fully_connected(fc1, 500, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        logits = tf.contrib.layers.fully_connected(fc2, 2, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        predictions = tf.nn.softmax(logits)
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(label, 1)),tf.float32))

        tf.global_variables_initializer().run()

        dict = np.load('./model/step_100_0806PM_April_04_2018.npy').item()
        for var in tf.global_variables():
            dict_name = var.name
            if dict_name in dict:
                print('loaded var: %s' % dict_name)
                sess.run(var.assign(dict[dict_name]))
            else:
                print('missing var %s, skipped' % dict_name)

        # print(sess.run(accuracy, feed_dict={data_ph: data_test, label_ph: label_test}))
        final_click = sess.run(predictions, feed_dict={data_ph: test})
        train_predict_click = sess.run(predictions, feed_dict={data_ph: train})
        val_predict_click = sess.run(predictions, feed_dict={data_ph: val})
    print(final_click.shape)
    print(test.shape)

    test_click = pd.DataFrame(final_click)
    test_click = pd.concat([test_click,test_slotprice],axis=1)
    test_click.to_csv('/Users/mengwei.zhang/Desktop/test1_click.csv', index = False)

    train_predict_click = pd.DataFrame(train_predict_click)
    train1 = pd.concat([train_predict_click,train_slotprice,train_payprice,train_bidprice],axis=1)
    train1.to_csv('/Users/mengwei.zhang/Desktop/train1_click.csv', index = False)

    val_predict_click = pd.DataFrame(val_predict_click)
    val1 = pd.concat([val_predict_click,val_slotprice,val_payprice,val_bidprice],axis=1)
    val1.to_csv('/Users/mengwei.zhang/Desktop/val1_click.csv', index = False)

if __name__ == "__main__":
    main()
