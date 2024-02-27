# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:30:37 2023

@author: urmii
"""

'''Problem Statement: -

Q) Build a recommender system with the given data using UBCF.

This dataset is related to the video gaming industry and a survey was conducted to build a 
recommendation engine so that the store can improve the sales of its gaming DVDs. Snapshot of the dataset is given below. Build a Recommendation Engine and suggest top selling DVDs to the store customers.

1.	Business Problem
       1.1.	What is the business objective?
       1.2.	Are there any constraints?'''
'''
Answer     1. Business objective
                  Maximize= saling the more number of gaming DVD's'
                  Minimize= error while recommendeing the gaming DVD's'
                  Constraints= improve accuracy while recommedeing and predicting the DVD's'
'''
'''
2.Work on each feature of the dataset to create a data 
dictionary as displayed in the image below:
    
Name of feature         description                  type             relevance
    userID            Identity of user             Discrete        ID is not useful(irrelevant)    
    game              name of games                Nominal           relevant
    rating            rating of each games        Continuous         relevant
'''



import pandas as pd
game=pd.read_csv('c:/2-Datasets/game.csv')
game
game.columns
#Index(['userId', 'game', 'rating'], dtype='object')
game.shape
#(5000, 3)
game.dtypes()

game.describe()
'''
userId       rating
count  5000.000000  5000.000000
mean   3432.282200     3.592500
std    1992.000866     0.994933
min       1.000000     0.500000
25%    1742.500000     3.000000
50%    3395.000000     4.000000
75%    5057.750000     4.000000
max    7120.000000     5.000000
'''

game.drop(['userId'],axis=1,inplace=True)
game.columns
#Index(['game', 'rating'], dtype='object')

game.describe()
'''rating
count  5000.000000
mean      3.592500
std       0.994933
min       0.500000
25%       3.000000
50%       4.000000
75%       4.000000
max       5.000000
'''

game.duplicated().sum()
#429
game.drop_duplicates(inplace=True)

game.duplicated().sum()
#0

game.isnull().sum()
'''game      0
rating    0
dtype: int64
'''

game1=pd.get_dummies(game)

game1.columns
game1.drop(['game_flower'],axis=1,inplace=True)
game1.columns
game1.describe()
game_df=game1.describe()

#we need to normalize data from data
#because data is not normalize if we check the rating and other columns 
# we need to make all entries in the 0 and 1 form 
def norm_func(i):
    x=(i-i.min()/i.max()-i.min())
    return x
df_norm=norm_func(game1)
b=df_norm.describe()
b
import seaborn as sns
#checking the outlier
sns.boxplot(df_norm)

#cdf pdf
#??

#######################Recommendation System################################
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')
game1.shape
#you will get 4571 X 3438 matrix
game['rating'].isnull().sum()

#now let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(game)
tfidf_matrix.shape
tfidf_matrix

from sklearn.metrics.pairwise import linear_kernel#This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)

#to get recommendation
import numpy as np
def get_recommendations(Name,topN):
    sorted_game=game.sort_values(by='rating',ascending=False)
    top_game=sorted_game.head(topN)
    #while assigning values,it is by default capturing original index of the
    #we want to reset the index
    top_game.reset_index(inplace=True,drop=True)
    print(top_game)
#enter your anime and number of animes to be recommended
get_recommendations('NASCAR Heat', topN=10)
    
    
    

   





















