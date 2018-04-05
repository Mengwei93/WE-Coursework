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


    train_bidprice = train0.bidprice
    train_payprice = train0.payprice
    train = train0.drop(['bidprice', 'payprice'], axis = 1)
    train_norm = (train - train.mean()) / (train.max() - train.min())

    val_bidprice = val0.bidprice
    val_payprice = val0.payprice
    val = val0.drop(['bidprice', 'payprice'], axis = 1)
    val_norm = (val - val.mean()) / (val.max() - val.min())

    test = test0
    test_norm = (test - test.mean()) / (test.max() - test.min())


    # all_id = [idx for idx in range(0, 96)]
    # label_id = [0]
    # price_id = [12]
    # exclude_id = [0, 3, 12]
    # attr_id = [idx for idx in all_id if idx not in set(exclude_id)]

    # all_train = np.load("./simple_npy/train2.npy")
    # data_train = all_train[:, attr_id]
    # price_train = all_train[:, price_id] / 100.0
    # data_train_scale = preprocessing.scale(data_train)

    # all_validate = np.load("./simple_npy/validation2.npy")
    # data_validate = all_validate[:, attr_id]
    # price_validate = all_validate[:, price_id] / 100.0
    # data_validate_scale = preprocessing.scale(data_validate)

    max_iter = 100

    with tf.Session() as sess:
        data_ph = tf.placeholder(tf.float32, [None, 3])
        label_ph = tf.placeholder(tf.int32, [None])
        label = tf.one_hot(label_ph, 2)
        fc1 = tf.contrib.layers.fully_connected(data_ph, 500, activation_fn=tf.nn.relu,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        fc2 = tf.contrib.layers.fully_connected(fc1, 500, activation_fn=tf.nn.relu,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        logits = tf.contrib.layers.fully_connected(fc2, 2, activation_fn=None,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        predictions = tf.nn.softmax(logits)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
        reg_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(label, 1)), tf.float32))

        total_loss = ce_loss + reg_loss

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.1, global_step,
                                                   100, 0.9, staircase=False)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


        tf.summary.scalar('cross_entropy', ce_loss)
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter('./log/')

        tf.global_variables_initializer().run()

        for i in range(max_iter):
            sess.run(train_step, feed_dict={data_ph: train_norm, label_ph: train_payprice})
            loss1, loss2, acc, summary = sess.run([ce_loss, reg_loss, accuracy, merged], feed_dict={data_ph: val_norm, label_ph: val_payprice})
            print("Iteration: [%4d/%4d], ce_loss: %.4f, reg_loss: %.4f, accuracy: %.4f" % (i, max_iter, loss1, loss2, acc))
            test_writer.add_summary(summary, i)

        string_time = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")

        str_temp = './model/price_step_100_' + string_time + '.npy'
        save_dict = {var.name: var.eval(sess) for var in tf.global_variables()}
        np.save(str_temp, save_dict)
        # data_ph = tf.placeholder(tf.float32, [None, 3])
        # price_ph = tf.placeholder(tf.float32, [None, 1])
        # fc1 = tf.contrib.layers.fully_connected(data_ph, 500, activation_fn=tf.nn.relu,
        #                                         weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        # fc2 = tf.contrib.layers.fully_connected(fc1, 500, activation_fn=tf.nn.relu,
        #                                         weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        # predictions = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=None,
        #                                                 weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        # mse_loss = tf.reduce_mean(tf.squared_difference(predictions, price_ph))
        # # reg_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total_loss = mse_loss

        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(0.1, global_step,
        #                                            100, 0.9, staircase=False)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

        # tf.summary.scalar('cross_entropy', ce_loss)
        # tf.summary.scalar('accuracy', accuracy)
        # merged = tf.summary.merge_all()
        # test_writer = tf.summary.FileWriter('./log/')

        # tf.global_variables_initializer().run()
        # count = 0

        # batch_size = 100000
        # batch_iter = len(data_train_scale) // batch_size

        # for i in range(max_iter):

        #     if i % batch_iter == 0:
        #         shuffle_id = np.arange(0, len(data_train_scale))
        #         np.random.shuffle(shuffle_id)
        #         count = 0

        #     data_train_batch = data_train_scale[shuffle_id[count * batch_size:(count + 1) * batch_size]]
        #     data_price_batch = price_train[shuffle_id[count * batch_size:(count + 1) * batch_size]]
        #     sess.run(train_step, feed_dict={data_ph: data_train_batch, price_ph: data_price_batch})
        #     count = count + 1
        #     if i % 50 == 0:
        #         loss_train = sess.run(mse_loss,
        #                               feed_dict={data_ph: data_train_batch,
        #                                          price_ph: data_price_batch})

        #         price_value, loss_validate = sess.run([predictions, mse_loss],
        #                                               feed_dict={data_ph: data_validate_scale,
        #                                                          price_ph: price_validate})
        #         print("Iteration: [%4d/%4d], mse_loss_train: %.4f" % (
        #             i, max_iter, loss_train))

        #         print("Iteration: [%4d/%4d], mse_loss_validate: %.4f" % (
        #             i, max_iter, loss_validate))

        #     # test_writer.add_summary(summary, i)

        # string_time = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")

        # str_temp = './model/step_10000_' + string_time + '.npy'
        # save_dict = {var.name: var.eval(sess) for var in tf.global_variables()}
        # np.save(str_temp, save_dict)

        # dict = np.load('/homes/kp306/Documents/project/cifar/checkpoint_cifar_gn_v1/cifar_100_gn_v1_153_0330.npy').item()
        # for var in tf.all_variables():
        #     dict_name = var.name
        #     if dict.has_key(dict_name):
        #         print('loaded var: %s' % dict_name)
        #         self.sess.run(var.assign(dict[dict_name]))
        #     else:
        #         print('missing var %s, skipped' % dict_name)


if __name__ == "__main__":
    main()
