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
############ drop irrelevant columns ###############
def drop_cols():
    column = [ 'click','weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility','slotformat', 'slotprice', 'bidprice','payprice', 'keypage', 'advertiser','usertag']
    train = pd.read_csv("G:/Business Analytics/Web Economics/train.csv")
    validation = pd.read_csv("G:/Business Analytics/Web Economics/validation.csv")
    test = pd.read_csv("G:/Business Analytics/Web Economics/test1.csv")
    
    train1 = train[column]
    validation1 = validation[column]
    test1 = test[[ 'weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility','slotformat', 'slotprice',  'keypage', 'advertiser','usertag']]
    
    train1.to_csv("G:/Business Analytics/Web Economics/code for we/train.csv",index = False)
    validation1.to_csv("G:/Business Analytics/Web Economics/code for we/validation.csv",index = False)
    test1.to_csv("G:/Business Analytics/Web Economics/code for we/test.csv",index = False)
    return 


def negative_down_sampling(train,ratio):
    pos = train[train.click == 1]
    neg = train[train.click == 0]
    neg = neg.sample(frac = ratio)
    a = pos.append(neg)
    a = a.sample(frac = 1)
    a = np.array(a)
    a = pd.DataFrame(a,columns = train.columns)
    return a
#a = pd.read_csv("G:/Business Analytics/Web Economics/code for we/train_neg_0.025.csv")
#a.to_csv("G:/Business Analytics/Web Economics/code for we/train_neg_0.025.csv",index = False)
def encoding1(XX): ## to encode the test dataset
    X = XX
    X = pd.concat([X,pd.get_dummies(X.weekday,prefix='day')],axis=1)
    X = X.drop('weekday',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.hour,prefix='hour')],axis=1)
    X = X.drop('hour',axis=1)
    
    df = pd.DataFrame(X.useragent.str.split('_',1).tolist(),columns = ['OS','browser'])
    X = pd.concat([X,df],axis=1)
    X = pd.concat([X,pd.get_dummies(X.OS,prefix='OS')],axis=1)
    X = X.drop('OS',axis=1)
    X = pd.concat([X,pd.get_dummies(X.browser,prefix='browser')],axis=1)
    X = X.drop('browser',axis=1)
    X = X.drop('useragent',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.region,prefix='region')],axis=1)
    X = X.drop('region',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.city,prefix='city')],axis=1)
    X = X.drop('city',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.adexchange,prefix='adexchange')],axis=1)
    X = X.drop('adexchange',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotwidth,prefix='slotwidth')],axis=1)
    X = X.drop('slotwidth',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotheight,prefix='slotheight')],axis=1)
    X = X.drop('slotheight',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotvisibility,prefix='slotvisibility')],axis=1)
    X = X.drop('slotvisibility',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotformat,prefix='slotformat')],axis=1)
    X = X.drop('slotformat',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.keypage,prefix='keypage')],axis=1)
    X = X.drop('keypage',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.advertiser,prefix='advertiser')],axis=1)
    X = X.drop('advertiser',axis=1)
    
    # X = X.drop('slotprice',axis=1)
    
    
    a = pd.DataFrame(X.usertag.str.split(',').tolist())
    usertag_df = pd.DataFrame(a)
    usertag_df2 = pd.get_dummies(usertag_df,prefix='usertag')
    usertag_df2 = usertag_df2.groupby(usertag_df2.columns, axis=1).sum()
    X = pd.concat([X, usertag_df2], axis=1)
    X = X.drop('usertag', axis=1)
    
    return X

def encoding(XX):## to encode the train and validation dataset
    X = XX
    X = pd.concat([X,pd.get_dummies(X.weekday,prefix='day')],axis=1)
    X = X.drop('weekday',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.hour,prefix='hour')],axis=1)
    X = X.drop('hour',axis=1)
    
    df = pd.DataFrame(X.useragent.str.split('_',1).tolist(),columns = ['OS','browser'])
    X = pd.concat([X,df],axis=1)
    X = pd.concat([X,pd.get_dummies(X.OS,prefix='OS')],axis=1)
    X = X.drop('OS',axis=1)
    X = pd.concat([X,pd.get_dummies(X.browser,prefix='browser')],axis=1)
    X = X.drop('browser',axis=1)
    X = X.drop('useragent',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.region,prefix='region')],axis=1)
    X = X.drop('region',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.city,prefix='city')],axis=1)
    X = X.drop('city',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.adexchange,prefix='adexchange')],axis=1)
    X = X.drop('adexchange',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotwidth,prefix='slotwidth')],axis=1)
    X = X.drop('slotwidth',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotheight,prefix='slotheight')],axis=1)
    X = X.drop('slotheight',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotvisibility,prefix='slotvisibility')],axis=1)
    X = X.drop('slotvisibility',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotformat,prefix='slotformat')],axis=1)
    X = X.drop('slotformat',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.keypage,prefix='keypage')],axis=1)
    X = X.drop('keypage',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.advertiser,prefix='advertiser')],axis=1)
    X = X.drop('advertiser',axis=1)
    
    # X = X.drop('slotprice',axis=1)
    # X = X.drop('bidprice',axis=1)
    # X = X.drop('payprice',axis=1)
    
    a = pd.DataFrame(X.usertag.str.split(',').tolist())
    usertag_df = pd.DataFrame(a)
    usertag_df2 = pd.get_dummies(usertag_df,prefix='usertag')
    usertag_df2 = usertag_df2.groupby(usertag_df2.columns, axis=1).sum()
    X = pd.concat([X, usertag_df2], axis=1)
    X = X.drop('usertag', axis=1)
    
    return X


def recal(p,w):
    return p/(p + (1-p)/w)