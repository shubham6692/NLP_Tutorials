# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:33:30 2020

@author: sagarw39
"""

import pandas as pd

messages=pd.read_csv("SMSSpamCollection", sep='\t', names=["label","message"])

# Tokenization of paragraphs/sentences
import nltk


               
#Cleaning the text
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
corpus=[]
for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages["message"][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words("English"))]
    review=' '.join(review)
    corpus.append(review)

# Creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=2500) # out of 6296, we will take top 2500 frequent
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages["label"])
y=y.iloc[:,1].values

# Train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X, y, test_size=.2, random_state=0)

# training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spm_model=MultinomialNB().fit(X_train,y_train)

y_pred=spm_model.predict(X_test)

from sklearn.metrics import confusion_matrix

conf=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,y_pred)
