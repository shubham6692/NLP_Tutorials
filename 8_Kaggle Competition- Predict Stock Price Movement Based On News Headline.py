# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:56:35 2020

@author: sagarw39
"""

import pandas as pd
df= pd.read_csv("Data_Stock_Sentiment.csv", encoding="ISO-8859-1")

# 1 means price increased
# 0 means same or decrease

train=df[df["Date"]<'20150101']
test=df[df["Date"]>'20141231']

# Remove puntuations
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Renaming column name
list1=[i for i in range(25)]
new_index=[str(i) for i in list1]
data.columns=new_index

# convert lower
for index in new_index:
    data[index]=data[index].str.lower()
    
# join 15 headline into 1

headlines=[]
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

#BOG
countvector=CountVectorizer(ngram_range=(2,2))
train_dataset=countvector.fit_transform(headlines)

#Random forest
random=RandomForestClassifier(n_estimators=200, criterion='entropy')
random.fit(train_dataset, train["Label"])

#same thing for test data
test_transform=[]
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=countvector.transform(test_transform)
pred=random.predict(test_dataset)

from sklearn.metrics import accuracy_score, confusion_matrix

cf=confusion_matrix(test["Label"],pred )
acc=accuracy_score(test["Label"],pred )



    




