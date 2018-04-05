import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from data_process import *

raw_train = pd.read_csv("/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/OneDrive_2018-02-25/train.csv")
raw_validation = pd.read_csv("/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/OneDrive_2018-02-25/validation.csv")
raw_test = pd.read_csv("/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/OneDrive_2018-02-25/test.csv")

column = [ 'click','weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility','slotformat', 'slotprice', 'bidprice','payprice', 'keypage', 'advertiser','usertag']

train = raw_train[column]
validation = raw_validation[column]
test = raw_test[[ 'weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility','slotformat', 'slotprice',  'keypage', 'advertiser','usertag']]

train = negative_down_sampling(train, 0.025)
validation = negative_down_sampling(validation, 0.1)
# test = negative_down_sampling(test, 0.025)
# print(train)
# print(train.head(),test.head(),validation.head())
train = encoding(train)
validation = encoding(validation)
test = encoding1(test)

print(train.shape, test.shape, validation.shape)

train.to_csv('/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/down_sample/train3.csv',index = False)
validation.to_csv('/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/down_sample/validation3.csv', index = False)
test.to_csv('/Users/mengwei.zhang/Desktop/COMPGW02 WE/Couesework/data/down_sample/test3.csv', index = False)