import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load Data
df1 = pd.read_csv('tmdb_5000_credits.csv', engine="python")
df2 = pd.read_csv('tmdb_5000_movies.csv')
df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')

# Load Ratings Data for Collaborative Filtering
ratings = pd.read_csv('ratings_small.csv')

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]

# Collaborative Filtering using SVD
def collaborative_filtering(user_id):
    # Preparing the data for Surprise library
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Splitting data into train and test set
    trainset, testset = train_test_split(data, test_size=0.25)
    
    # Using SVD for collaborative filtering
    svd = SVD()
    svd.fit(trainset)
    
    # Predict ratings for all movies not yet rated by the user
    user_ratings = ratings[ratings['userId'] == int(user_id)]
    user_rated_movies = user_ratings['movieId'].tolist()
    all_movie_ids = ratings['movieId'].unique()
    movie_ids_to_predict = np.setdiff1d(all_movie_ids, user_rated_movies)
    
    predictions = [svd.predict(user_id, movie_id) for movie_id in movie_ids_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_movie_ids = [int(pred.iid) for pred in predictions[:10]]
    top_movies = df2[df2['id'].isin(top_movie_ids)]['title'].tolist()
    
    return top_movies

# Streamlit Interface
st.title('Movie Recommendation System')

option = st.selectbox('Choose a recommendation type:',
                      ('Content-Based', 'Collaborative Filtering'))

if option == 'Content-Based':
    movie = st.text_input('Enter a movie title:')
    if st.button('Recommend'):
        recommendations = get_recommendations(movie)
        st.write(recommendations)

elif option == 'Collaborative Filtering':
    user_id = st.text_input('Enter your User ID:')
    if st.button('Recommend'):
        recommendations = collaborative_filtering(user_id)
        st.write(recommendations)
