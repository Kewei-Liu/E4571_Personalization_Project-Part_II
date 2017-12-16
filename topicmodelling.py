#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:59:06 2017

@author: rakshitanagalla
"""


# tf-idf book title
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re 
import string
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

train = final_matrix[['ISBN', u'summary']].drop_duplicates()

train.dropna(subset=['summary'], how='any',inplace=True)
train=train[train['summary']!='NA']
all_titles = train['summary']
X =[]
print('Tokenising....')
for title in all_titles:
    x = re.split(' ', title)
    X.append(' '.join(x)) 
    
lemmatiser = WordNetLemmatizer() 
print('Pre-processing...')    
for j in range(0,len(X)):
    #remove numbers
    X[j] = re.sub('[0-9]+','', X[j])
    #remove tags
    X[j] = re.sub('<i>|</i>|<br />|\\u','', X[j])
    #remove punctuations and convert to lower case
    X[j] = ''.join([i.lower() for i in X[j] if i not in string.punctuation])
    #remove non-ascii characters 
    X[j] = ''.join(i for i in X[j] if ord(i)<128)
    #remove stopwords
    X[j] = ' '.join([word for word in X[j].split() if word not in (stopwords.words('english'))])
    #lemmatize
    X[j] = ' '.join([lemmatiser.lemmatize(word) for word in X[j].split()])
    
train['summary']=X

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.5,
                             min_df=2, stop_words='english',
                             ngram_range=(1,2))
X_train_new = vectorizer.fit_transform(train['summary']) 

import lda

model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(X_train_new)

dtp = model.doc_topic_

#topic_word = model.topic_word_ 
#vocab=vectorizer.vocabulary_
#vocab2 = {y:x for x,y in vocab.iteritems()}
#
#
#n_top_words = 6
#for i, topic_dist in enumerate(topic_word):
#    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
#    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

lda_features = pd.DataFrame(dtp)
lda_features['ISBN'] = list(train['ISBN'])
past_features = pd.read_csv('w2v_features.csv')
final=pd.merge(past_features, lda_features, how='inner', on='ISBN')




final.drop(['Unnamed: 0','Book-Title'], axis=1,inplace=True)
final.drop(['Book-Title'], axis=1,inplace=True)

ratings = final_matrix[['User-ID', u'ISBN', u'Book-Rating']]

full = pd.merge(ratings, final, how='inner', on='ISBN')

msk = np.random.rand(len(full)) < 0.8
train=np.matrix(full[msk])
test = np.matrix(full[~msk])


from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute',n_neighbors=5)
#fit on training data
model_knn.fit(train[:,3:])
#compute 5 nearest neighbors for items in test
distances, indices = model_knn.kneighbors(test[:,3:])


predictions =[]
for i in range(len(test)):
    uid = test[i,0]
    iid = test[i,1]
    true_r = test[i,2]
    est_r = np.matrix(distances[i,:])*train[indices[i,:],2]
    #est_r = est_r + float(user_bias[user_bias['User-ID']==uid]['Book-Rating'])
    predictions.append((uid, iid,est_r[0,0],true_r))

print('With LDA features: '+str(np.mean([(tup[2]-tup[3])**2 for tup in predictions])))

    
 #####   
from sklearn.neighbors import NearestNeighbors
model_knn2 = NearestNeighbors(metric='cosine', algorithm='brute',n_neighbors=5)
#fit on training data
model_knn2.fit(train[:,3:-20])
#compute 5 nearest neighbors for items in test
distances, indices = model_knn2.kneighbors(test[:,3:-20])


predictions2 =[]
for i in range(len(test)):
    uid = test[i,0]
    iid = test[i,1]
    true_r = test[i,2]
    est_r = np.matrix(distances[i,:])*train[indices[i,:],2]
    #est_r = est_r + float(user_bias[user_bias['User-ID']==uid]['Book-Rating'])
    predictions2.append((uid, iid,est_r[0,0],true_r))

    
print('Without LDA features: '+str(np.mean([(tup[2]-tup[3])**2 for tup in predictions2])))
    
