# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:43:48 2022

@author: Neela
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    

#creating the bag of words model

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
pickle.dump(cv, open('tranform.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20 ,random_state=0)


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)
filename = 'nlpmine_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
