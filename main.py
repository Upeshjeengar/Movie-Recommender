import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load Data
df1 = pd.read_csv('https://firstbucket0125.s3.ap-south-1.amazonaws.com/data/tmdb_5000_credits.csv', engine="python")
df2 = pd.read_csv('tmdb_5000_movies.csv')
df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')

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


#  Collaborative Filtering using SVD
def collaborative_filtering(user_id):
    # Load Ratings Data for Collaborative Filtering
    pass


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
      #recommendations=collaborative_filtering(user_id)
      st.write("Feature yet to be implemented")
