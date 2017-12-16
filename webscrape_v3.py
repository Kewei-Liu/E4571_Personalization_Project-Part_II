# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:07:05 2017

@author: Deepak Maran
"""
import os
import pandas as pd
import numpy as np
import time

os.chdir('C:\Users\Deepak Maran\Projects\Personalization\E4571_Personalization_Project-Part_I-master')

users = pd.read_csv('BX-Users.csv',sep=';') #278858 users
items = pd.read_csv('BX-Books.csv',sep=';',error_bad_lines=False) #271360 items
ratings = pd.read_csv('BX-Book-Ratings.csv',sep=';') #105283 users/ 340556 items. More items in ratings than in items.
ratings = ratings[ratings['Book-Rating'] > 0] # Users: 77805 items: 185973

from goodreads import client
api_key = 'jOwrlz45qhGLOXKI1hkzVQ'
api_secret = 'vOP6rOQYv8P2AZgiF10NWtvk45XEEVRdtDI3rnhl00'
gc = client.GoodreadsClient(api_key, api_secret)


### Sparser Ratings Matrix
ratings_less_rated_only = ratings.groupby('ISBN').filter(lambda x: len(x) < 10)
print len(set(ratings_less_rated_only['ISBN']))
ratings_sampled = ratings_less_rated_only.sample(10000)

unique_isbns = set(ratings_sampled['ISBN'])
items_sampled = items[items['ISBN'].isin(unique_isbns)]
sLength = len(items_sampled['ISBN'])
items_sampled['summary'] = pd.Series(np.array(['NA']*sLength), index=items_sampled.index)

for index, item in items_sampled.iterrows():
    isbn = item['ISBN']
    title = item['Book-Title']
    try:
        book = gc.book(isbn=isbn)
        print book.title, index
        items_sampled['summary'][index] = book.description
    except:
        print 'No summary for book:', isbn, title


### Denser Ratings Matrix
# b is the sample containing ratings with items with atleast 10 ratings 
# and users who rated 20 items

ratings = ratings[ratings['ISBN'].isin(items['ISBN'])] # Users: 68091 items: 149836
a = ratings.groupby('ISBN').filter(lambda x: len(x) >= 10)
b = a.groupby('User-ID').filter(lambda x: len(x) >= 20)
ratings_sampled_dense = b.sample(10000)

unique_isbns_dense = set(ratings_sampled_dense['ISBN'])
items_sampled_dense = items[items['ISBN'].isin(unique_isbns_dense)]
sLength = len(items_sampled_dense['ISBN'])
items_sampled_dense['summary'] = pd.Series(np.array(['NA']*sLength), index=items_sampled_dense.index)

i=0
for index, item in items_sampled_dense.iterrows():
    print i
    i=i+1
    isbn = item['ISBN']
    title = item['Book-Title']
    try:
        book = gc.book(isbn=isbn)
#        print book.title, index
        items_sampled_dense['summary'][index] = book.description
    except:
        print 'No summary for book:', isbn, title

# Concatenate Ratings matrices
frames_1 = [ratings_sampled, ratings_sampled_dense]
merged_ratings = pd.concat(frames_1)
# Concatenate Book Info
frames_2 = [items_sampled, items_sampled_dense]
merged_info = pd.concat(frames_2)


### Add book and user info columns to ratings matrix
ts = time.time()

final_sample = pd.merge(merged_ratings, merged_info, on = 'ISBN')
final_matrix = pd.merge(final_sample, users, on = 'User-ID')

print time.time() - ts

    
# final_matrix.to_csv('final_matrix.csv')

### Create train and test
# train = 80% of the data, test = 20% of the data
np.random.seed(seed = 111)
msk = np.random.rand(len(final_matrix)) < 0.8
train = final_matrix[msk]
test = final_matrix[~msk]
