#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:45:59 2020

@author: swapnilchauhan
"""
# Importing all the libraries
import nltk





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from string import punctuation 
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report,f1_score
import lightgbm as lgb
from sklearn.naive_bayes import  BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# IMPORTING TRAIN AND TEST DATA 

train=pd.read_csv('train-3.csv')
test=pd.read_csv('test-3.csv')

test1=test

train.head()

test.head()

train['target'].value_counts()

train.shape

train.isnull().sum()

test.shape

test.isnull().sum()

data=pd.concat([train,test],sort=False,ignore_index=True)

data


data=data[['text','target','location']]

data['location']=data['location'].fillna(method='bfill')
data['location']=data['location'].fillna(method='ffill')
data['location'].isnull().sum()

lr=LabelEncoder()
data['location']=lr.fit_transform(data['location'])



data.shape

data.isnull().sum()

data

for i in range(0,10876):
    data['text'][i]=data['text'][i].lower()
    data['text'][i]=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',data['text'][i])
    data['text'][i]=re.sub('@[^\s]+','USER',data['text'][i])
    data['text'][i]=re.sub(r'#([^\s]+)',r'\1',data['text'][i])


unw=set(stopwords.words('english')+list(punctuation))

stemmer = LancasterStemmer()

corpus=data['text']

corpus

final_corpus= []
for i in range(len(corpus)):
    text = word_tokenize(corpus[i].lower())
    text = [stemmer.stem(x) for x in text if x not in unw]
    final = " ".join(text)
    final_corpus.append(final)

#Apply tfidf

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(final_corpus).toarray()
y= train['target']

X=pd.DataFrame(X)

data=pd.concat([data,X],axis=1)

data=data.drop('text',axis=1)

train=data[data['target'].notnull()]
test=data[data['target'].isnull()]

X=train.drop(['target'],axis=1)

X.shape

#Split Dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred4=classifier.predict(X_train)
y_pred5=classifier.predict(X_test)
y_prob4=classifier.predict_proba(X_train)[:,1]
y_prob5=classifier.predict_proba(X_test)[:,1]

print(confusion_matrix(y_train, y_pred4))

print(confusion_matrix(y_test, y_pred5))


print(classification_report(y_test, y_pred5))

print(classification_report(y_train, y_pred4))

print(f1_score(y_train,y_pred4))

print(f1_score(y_test,y_pred5))

# RF

rr=RandomForestClassifier()

rr.fit(X_train,y_train)

y_pred1=rr.predict(X_train)
y_pred2=rr.predict(X_test)


print(confusion_matrix(y_train, y_pred1))


print(confusion_matrix(y_test, y_pred2))

print(f1_score(y_train,y_pred1))

print(f1_score(y_test,y_pred2))

print(classification_report(y_test, y_pred2))

print(classification_report(y_train, y_pred1))

#RF HYPERTUNED 


rf=RandomForestClassifier()
params= {
    'n_estimators':sp_randint(240,300),
    'max_depth':sp_randint(50,200),
    
}

rr=RandomizedSearchCV(estimator=rf,param_distributions=params,cv=5,scoring='roc_auc',random_state=1)
rr.fit(X_train,y_train)

rf=RandomForestClassifier(**rr.best_params_)
rf.fit(X_train,y_train)

ytrain_pred3=rf.predict(X_train)
ytest_pred3=rf.predict(X_test)

print(confusion_matrix(y_train, ytrain_pred3))

print(confusion_matrix(y_test, ytest_pred3))


print(f1_score(y_train,ytrain_pred3))

print(f1_score(y_test,ytest_pred3))

print(classification_report(y_test, ytest_pred3))

print(classification_report(y_train, ytrain_pred3))


#XGBOOST CLASSIFIER

xg=XGBClassifier(max_depth=10,learning_rate=0.5)
xg.fit(X_train,y_train)

ytrain_pred6=xg.predict(X_train)
ytest_pred6=xg.predict(X_test)

print(confusion_matrix(y_train, ytrain_pred6))

print(confusion_matrix(y_test, ytest_pred6))


print(f1_score(y_train, ytrain_pred6))

print(f1_score(y_test, ytest_pred6))

print(classification_report(y_train, ytrain_pred6))

print(classification_report(y_test, ytest_pred6))

#WITH DIFF PARAMS
xg1=XGBClassifier(learning_rate=1,n_estimators=500)
xg1.fit(X_train,y_train)

ytrain_pred7=xg1.predict(X_train)
ytrain_prob7=xg1.predict_proba(X_train)[:,1]
ytest_pred7=xg1.predict(X_test)
ytest_prob7=xg1.predict_proba(X_test)[:,1]

print(confusion_matrix(y_train, ytrain_pred7))

print(confusion_matrix(y_test, ytest_pred7))


print(f1_score(y_train, ytrain_pred7))

print(f1_score(y_test, ytest_pred7))



print(classification_report(y_train, ytrain_pred7))

print(classification_report(y_test, ytest_pred7))


# LIGHT GBM

lgbmc=lgb.LGBMClassifier(random_state=1)
lgbmc.fit(X_train,y_train)

y_train_pred9=lgbmc.predict(X_train)
y_test_pred9=lgbmc.predict(X_test)


print('Confusion Matrix:','\n',confusion_matrix(y_train,y_train_pred9))

print('Confusion Matrix:','\n',confusion_matrix(y_test,y_test_pred9))



print(f1_score(y_train,y_train_pred9))

print(f1_score(y_test,y_test_pred9))



print(classification_report(y_train,y_train_pred9))

print(classification_report(y_test,y_test_pred9))




bnb=BernoulliNB(alpha=1.2)
bnb.fit(X_train,y_train)


y_train_pred7=bnb.predict(X_train)
y_test_pred7=bnb.predict(X_test)

print('Confusion Matrix:','\n',confusion_matrix(y_train,y_train_pred7))


print('Confusion Matrix:','\n',confusion_matrix(y_test,y_test_pred7))


print(f1_score(y_train,y_train_pred7))

print(f1_score(y_test,y_test_pred7))



print(classification_report(y_train,y_train_pred7))

print(classification_report(y_test,y_test_pred7))


# ADDABOOST CLASSIFIER

rf=RandomForestClassifier()

adac=AdaBoostClassifier(random_state=1,base_estimator=rf)
adac.fit(X_train,y_train)

y_train_pred8=adac.predict(X_train)
y_train_prob8=adac.predict_proba(X_train)[:,1]
y_test_pred8=adac.predict(X_test)
y_test_prob8=adac.predict_proba(X_test)[:,1]

print('Confusion Matrix:','\n',confusion_matrix(y_train,y_train_pred8))


print('Confusion Matrix:','\n',confusion_matrix(y_test,y_test_pred8))


print(f1_score(y_train,y_train_pred8))

print(f1_score(y_test,y_test_pred8))



print(classification_report(y_train,y_train_pred8))

print(classification_report(y_test,y_test_pred8))





lgbmc=lgb.LGBMClassifier(random_state=1,min_child_samples=15)
lgbmc.fit(X_train,y_train)

y_train_pred9=lgbmc.predict(X_train)
y_train_prob9=lgbmc.predict_proba(X_train)[:,1]
y_test_pred9=lgbmc.predict(X_test)
y_test_prob9=lgbmc.predict_proba(X_test)[:,1]




print('Confusion Matrix:','\n',confusion_matrix(y_train,y_train_pred9))
accuracy_score(y_train,y_train_pred9),roc_auc_score(y_train,y_train_prob9)

print('Confusion Matrix:','\n',confusion_matrix(y_test,y_test_pred9))
accuracy_score(y_test,y_test_pred9),roc_auc_score(y_test,y_test_prob9)
