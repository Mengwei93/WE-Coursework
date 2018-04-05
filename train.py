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
    # all_id = [idx for idx in range(0, 96)]
    # label_id = [0]
    # price_id = [12]
    # exclude_id = [0, 3, 12]
    # attr_id = [idx for idx in all_id if idx not in set(exclude_id)]

    # all_train = np.load("./simple_npy/train2.npy")
    # data_train = all_train[:, attr_id]
    # label_train = all_train[:, label_id]
    # data_train_scale = preprocessing.scale(data_train)

    # pos_train_id = np.where(label_train == 1)[0]
    # neg_train_id = np.where(label_train == 0)[0]

    # factor = len(neg_train_id) // len(pos_train_id)

    # data_pos_train = data_train_scale[pos_train_id]
    # label_pos_train = label_train[pos_train_id, 0]
    # data_neg_train = data_train_scale[neg_train_id]
    # label_neg_train = label_train[neg_train_id, 0]

    # data_pos_train = np.tile(data_pos_train, [factor, 1])
    # label_pos_train = np.tile(label_pos_train, [factor])

    # data_combine = np.concatenate((data_pos_train, data_neg_train), axis=0)
    # label_combine = np.concatenate((label_pos_train, label_neg_train), axis=0)

    # import pdb
    # pdb.set_trace()

    # all_validate = np.load("./simple_npy/validation2.npy")
    # data_validate = all_validate[:, attr_id]
    # label_validate = all_validate[:, label_id]
    # data_validate_scale = preprocessing.scale(data_validate)

    # pos_validate_id = np.where(label_validate == 1)[0]
    # neg_validate_id = np.where(label_validate == 0)[0]

    # data_pos_validate = data_validate_scale[pos_validate_id]
    # label_pos_validate = label_validate[pos_validate_id, 0]

    # data_neg_validate = data_validate_scale[neg_validate_id]
    # label_neg_validate = label_validate[neg_validate_id, 0]

    max_iter = 100

    with tf.Session() as sess:
        data_ph = tf.placeholder(tf.float32, [None, 593])
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

        # tf.summary.scalar('cross_entropy', ce_loss)
        # tf.summary.scalar('accuracy', accuracy)
        # merged = tf.summary.merge_all()
        # test_writer = tf.summary.FileWriter('./log/')

        # tf.global_variables_initializer().run()

        tf.summary.scalar('cross_entropy', ce_loss)
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter('./log/')

        tf.global_variables_initializer().run()

        for i in range(max_iter):
            sess.run(train_step, feed_dict={data_ph: train, label_ph: train_click})
            loss1, loss2, acc, summary = sess.run([ce_loss, reg_loss, accuracy, merged], feed_dict={data_ph: val, label_ph: val_click})
            print("Iteration: [%4d/%4d], ce_loss: %.4f, reg_loss: %.4f, accuracy: %.4f" % (i, max_iter, loss1, loss2, acc))
            test_writer.add_summary(summary, i)

        string_time = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")

        str_temp = './model/step_100_' + string_time + '.npy'
        save_dict = {var.name: var.eval(sess) for var in tf.global_variables()}
        np.save(str_temp, save_dict)

        # count = 0

        # batch_size = 100000


        # batch_iter = len(data_combine) // batch_size

        # for i in range(max_iter):

        #     if i % batch_iter == 0:
        #         shuffle_id = np.arange(0, len(data_combine))
        #         np.random.shuffle(shuffle_id)
        #         count = 0

        #     data_train_batch = data_combine[shuffle_id[count * batch_size:(count + 1) * batch_size]]
        #     data_label_batch = label_combine[shuffle_id[count * batch_size:(count + 1) * batch_size]]
        #     sess.run(train_step, feed_dict={data_ph: data_train_batch, label_ph: data_label_batch})
        #     count = count + 1
        #     if i % 100 == 0:
        #         pos_loss1, pos_loss2, acc_pos = sess.run([ce_loss, reg_loss, accuracy],
        #                                                  feed_dict={data_ph: data_pos_validate,
        #                                                             label_ph: label_pos_validate})

        #         neg_loss1, neg_loss2, acc_neg = sess.run([ce_loss, reg_loss, accuracy],
        #                                                  feed_dict={data_ph: data_neg_validate,
        #                                                             label_ph: label_neg_validate})

        #         print("Iteration: [%4d/%4d], ce_loss: %.4f, reg_loss: %.4f, accuracy_pos: %.4f" % (
        #             i, max_iter, pos_loss1, pos_loss2, acc_pos))
        #         print("Iteration: [%4d/%4d], ce_loss: %.4f, reg_loss: %.4f, accuracy_neg: %.4f" % (
        #             i, max_iter, neg_loss1, neg_loss2, acc_neg))

            # test_writer.add_summary(summary, i)

        # string_time = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")

        # str_temp = './model/step_50000_' + string_time + '.npy'
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
