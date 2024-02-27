# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:42:40 2023

@author: urmii
"""

'''Problem Statement: -

The Entertainment Company, which is an online movie watching platform, wants to improve its 
collection of movies and showcase those that are highly rated and recommend 
those movies to its customer by their movie watching footprint. For this, 
the company has collected the data and shared it with you to provide some 
analytical insights and also to come up with a recommendation algorithm 
so that it can automate its process for effective recommendations. 
The ratings are between -9 and +9.'''

'''1.	Business Problem
1.1.	What is the business objective?
1.1.	Are there any constraints?
'''

'''business objective:
    Maximize:increase the performance and accuracy of movies and recommend the movies
    according to user likes and rating
    Minimize: Reduce the error code while recommending the movies to user
    Constraints:improve performance and quality of movies
'''
'''2.Work on each feature of the dataset to create a data 
dictionary as displayed in the image below:
    
Name of feature         description                  type             relevance
    ID                Identity of user             Discrete        ID is not useful(irrelevant)    
    Titles            Name of movies               Nominal           relevant
    Category          type of movies               Nominal           relevant
    Reviews           Rating of movies            Continuous         relevant
'''
import pandas as pd
movies=pd.read_csv('c:/2-Datasets/Entertainment.csv')
movies
movies.columns
#Index(['Id', 'Titles', 'Category', 'Reviews'], dtype='object')
movies.shape
#(51,4)
movies.dtypes()
movies.describe()
'''         Id        Reviews
count    51.000000  51.000000
mean   6351.196078  36.289608
std    2619.679263  49.035042
min    1110.000000  -9.420000
25%    5295.500000  -4.295000
50%    6778.000000   5.920000
75%    8223.500000  99.000000
max    9979.000000  99.000000
'''
movies.duplicated().sum()
#429
movies.drop_duplicates(inplace=True)

movies.duplicated().sum()
#0

movies.isnull().sum()
'''Id          0
Titles      0
Category    0
Reviews     0
dtype: int64
'''
movies.drop(['Id'],axis=1,inplace=True)
movies.columns
#Index(['Titles', 'Category', 'Reviews'], dtype='object')
movies

movies.duplicated().sum()

movies.drop_duplicates(inplace=True)

movies.duplicated().sum()
#0
movies.isnull().sum()
'''Titles      0
Category    0
Reviews     0
dtype: int64
'''
movies.describe()


#here we are considering only genre
from sklearn.feature_extraction.text import TfidfVectorizer
#this is term freq inverse documents
#each row is treated as documents
tfidf=TfidfVectorizer(stop_words='english')
#it is going to create tfidfverctorizer to seperate all stop words
#it is going to seperate
#out all words from the row
#now let us check is there any null value
movies['Category']=movies['Category'].fillna('Category')
#now let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(movies.Category)
tfidf_matrix.shape
tfidf_matrix
#you will get 51,34
#it has created sparse matrix it means
#that we have 51 category
#on this particular matrix
#we want to do item based recommendation if a user has
#watched gadar then you reommend shershah movie
from sklearn.metrics.pairwise import linear_kernel
#this is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf_matrix similarity
#with each element of tfidf_matrix onlty
#output will be similiar matrix of size 51X34 size
#here is cosine_sim_matrix
#there are no movie names only index are provided
#we will try to map movie index given
#for that purpose custom function is written
movies_index = pd.Series(movies.index, index=movies['Titles']).drop_duplicates()
#we are converting anime_index into series format we want index and corresponding
movies_id=movies_index['Jumanji (1995)']
movies_id
def get_recommendations(Name,topN):
    #topN=10
    #name=Assassins (1995)
    movies_id=movies_index[Name]
    #we wany to capture whiole row of given movie
    #name its score and column id
    #for that purpose we are applying cosine_sim_matrix
    #to ensure function
    #enumerate function create a object
    #which we need to create in  list form
    #what enumerate does suppose we have given
    #(2,10,15,18) if we apply to enumerate then it will create a list
    #(0,2, 1,10, 3,15, 4,18)
    cosine_scores=list(enumerate(cosine_sim_matrix[movies_id]))
    #the cosine scores captures we want to arrange in descending order
    #sp that
    #we want recomment top 10 based on highest similarity i.e. score
    #if we will check the cosine score it comprises of index:cosine score
    #we want arrange tuples according to decrasing order
    #of the score not iindex
    #sorting the cosine_similarity scores based on socres i.e.x[1]
    cosine_scores=sorted(cosine_scores,key=lambda x:x[1],reverse=True)
    #get the score of top N most similar movies
    #to capture topN you need to give topN+1
    cosine_score_N=cosine_scores[0:topN+1]
    #getting the movie index
    movies_idx=[i[0] for i in cosine_score_N]
    #getting cosine score
    movies_scores=[i[1] for i in cosine_score_N]
    #we are going to use this information to create a dataframe
    #create a empty datafrme
    movies_similar_show=pd.DataFrame(columns=['name'])
    #assign anime_ind to name columns
    movies_similar_show['name']=movies.loc[movies_idx,'Titles']
    #assign score to score column
    #movies_similar_show['score']=movies_scores
    #while assinging values it it bu default capturing original
    #index of the
    #we want to reset the index
    movies_similar_show.reset_index(inplace=True)
    print(movies_similar_show)
    #enter your anime and number of animes to be recommended
get_recommendations('Jumanji (1995)', topN=10)
    
    
    





















































