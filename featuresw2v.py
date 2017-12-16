#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:01:46 2017

@author: rakshitanagalla
"""

import pandas as pd
import numpy as np

#items_sampled.dropna(axis=0, how='any',inplace=True)
#items_sampled.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1,inplace=True)
#items=items_sampled.copy() #<-- replace with final items file here

ratings = pd.read_csv('final.csv')
items = ratings[['ISBN', u'Book-Title',u'Book-Author', u'Year-Of-Publication', u'Publisher']].drop_duplicates()
#author, publisher - one hot encode
#one hot encoding publisher
items["Publisher"][items["Publisher"].replace(items["Publisher"].value_counts().to_dict()) < 5] = 'other_publisher'
items["Publisher"] = items["Publisher"].astype('category') #16808 publishers
items["Publisher"] = items["Publisher"].cat.codes

items["Book-Author"][items["Book-Author"].replace(items["Book-Author"].value_counts().to_dict()) < 5] = 'other_author'
items["Book-Author"] = items["Book-Author"].astype('category') #16808 publishers
items["Book-Author"] = items["Book-Author"].cat.codes

items_enc = pd.get_dummies(items, columns=["Publisher","Book-Author"])

# tf-idf book title
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re 
import string
from nltk.corpus import stopwords

all_titles = items_enc['Book-Title']
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
    #remove punctuations and convert to lower case
    X[j] = ''.join([i.lower() for i in X[j] if i not in string.punctuation])
    #remove non-ascii characters 
    X[j] = ''.join(i for i in X[j] if ord(i)<128)
    #remove stopwords
    X[j] = ' '.join([word for word in X[j].split() if word not in (stopwords.words('english'))])
    #lemmatize
    X[j] = ' '.join([lemmatiser.lemmatize(word) for word in X[j].split()])
    
items_enc['Book-Title']=X

#Only store vectors for words in our dataset          
all_words = set(word for title in X for word in title.split(' '))
#GLOVE_6B_50D_PATH = "glove.6B/glove.6B.300d.txt"
GLOVE_6B_50D_PATH = 'glove.42B.300d.txt'
glove_big = {}
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode("utf8").lower()
        nums = map(float, parts[1:])
        if word in all_words:
            glove_big[word] = list(nums)

#computes vector for each title by taking average of word vectors
def MeanVectorizer(word2vec, train):
    #TO DO: try tf-idf version 
    dim = len(list(glove_big.values())[0])
    return [
            np.mean([word2vec[w] for w in title.split(' ') if w in word2vec] 
                    or [np.zeros(dim)], axis=0)
            for title in train
        ]

X_new = MeanVectorizer(glove_big, X)
w2v_features = pd.DataFrame(X_new)
w2v_features['ISBN'] = list(items_enc['ISBN'])
items_enc=pd.merge(items_enc, w2v_features, how='inner', on='ISBN')

print('Out of the '+str(len(all_words))+' words in our corpus, '+str(len(glove_big.keys()))+' are in the pre-trained corpus')

items_enc.drop(['Year-Of-Publication'], axis=1,inplace=True)

items_enc.to_csv('w2v_features.csv')

## Train Test split

items_enc.drop(['Book-Title'], axis=1,inplace=True)

ratings = ratings[['User-ID', u'ISBN', u'Book-Rating']]


density = (float(len(ratings))/(len(np.unique(ratings['User-ID']))*len(np.unique(ratings['ISBN']))))*100
print "Density in percent: "+str(density) 
print "Users: "+str(len(np.unique(ratings['User-ID'])))+ " items: "+str(len(np.unique(ratings['ISBN'])))

full = pd.merge(ratings, items_enc, how='inner', on='ISBN')

msk = np.random.rand(len(full)) < 0.8

#user_bias = full[msk].groupby(['User-ID'])['Book-Rating'].mean().reset_index()
#x = pd.merge(full[msk], user_bias, how='inner', on='User-ID')
#x['Book-Rating_x']= x['Book-Rating_x']-x['Book-Rating_y']
#del x['Book-Rating_y']
#train = np.matrix(x)
full[msk].to_csv('train.csv')
full[~msk].to_csv('test.csv')