# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:00:34 2017

@author: Deepak Maran
"""

# Load final_matrix_merged.spydata, get 'final_matrix' variable

from collections import Counter
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
plt = matplotlib.pyplot

#isbns = ratings['ISBN']
#userids = ratings['User-ID']
#isbn_counts = Counter(isbns)
#userid_counts = Counter(userids)
#more_popular_isbns = []
#moderately_popular_isbns = []
#less_popular_isbns = []
#for key in isbn_counts.keys():
#    if isbn_counts[key] >= 2:
#        more_popular_isbns.append(key)
#    else:
#        less_popular_isbns.append(key)
#print len(more_popular_isbns)
#print len(less_popular_isbns)


### Item segmentation based on popularity
isbns = test['ISBN']
isbn_counts = Counter(isbns)
more_popular_isbns = []
less_popular_isbns = []
for key in isbn_counts.keys():
    if isbn_counts[key] >= 2:
        more_popular_isbns.append(key)
    else:
        less_popular_isbns.append(key)
print len(more_popular_isbns)
print len(less_popular_isbns)

ratings_more_popular_books = test[test['ISBN'].isin(more_popular_isbns)]
ratings_less_popular_books = test[test['ISBN'].isin(less_popular_isbns)]


ratings_per_book = isbn_counts.values()
plt.hist(ratings_per_book)
plt.xlabel('Number of times rated', size=28)
plt.ylabel('Number of books', size=28)
plt.title(r'Histogram of Number of ratings per Book')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


### User segmentation based on usage
userids = test['User-ID']
userid_counts = Counter(userids)
more_prolific_users = []
less_prolific_users = []
for key in userid_counts.keys():
    if userid_counts[key]>=2:
        more_prolific_users.append(key)
    else:
        less_prolific_users.append(key)
print len(more_prolific_users)
print len(less_prolific_users)

ratings_per_user = userid_counts.values()
plt.hist(ratings_per_user)
plt.xlabel('Number of times rated', size=28)
plt.ylabel('Number of users', size=28)
plt.title(r'Histogram of Number of ratings per User')
plt.grid(True)
plt.show()


### User segmentation based on age
userages = test['Age']
old_users = test[test['Age']>=40]
mid_users = test[(test['Age']>=30) & (test['Age']<50)]
young_users = test[test['Age']<30]
len(old_users)
len(mid_users)
len(young_users)

age = Counter(userages).values()
existing_userages = userages.dropna()
feasible_userages = existing_userages[existing_userages<200]
plt.hist(feasible_userages)
plt.xlabel('Age', size=28)
plt.ylabel('Number of users', size=28)
plt.title(r'Histogram of Age')
plt.grid(True)
plt.show()


### Item segmentation based on year of publication
book_ages = test['Year-Of-Publication']
new_books = test[test['Year-Of-Publication']>=2000]
mid_age_books = test[test['Year-Of-Publication']>=1995]
old_books = test[test['Year-Of-Publication']<1995]
len(new_books)
len(mid_age_books)
len(old_books)

book_age = Counter(book_ages).values()
existing_bookages = book_ages.dropna()
feasible_bookages = existing_bookages[(existing_bookages>1800) & (existing_bookages<2017)]
plt.hist(feasible_bookages, bins = 30)
plt.xlabel('Book Publication year', size=28)
plt.ylabel('Number of books', size=28)
plt.title(r'Histogram of Book Publication year')
plt.axis([1960, 2010, 0, 1000])
plt.grid(True)
plt.show()


