import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load Data
df1 = pd.read_csv('https://firstbucket0125.s3.ap-south-1.amazonaws.com/data/tmdb_5000_credits.csv?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDUaCmFwLXNvdXRoLTEiRzBFAiEA5OOHqGWxT5n%2BHZ5sFsPyEKCzLipVxmP9%2FQDsK%2Fe9AToCIGVXutVVB3cUzsZlwieT1pee9RFa4QqIWyutdLJCQh0zKuQCCC4QAhoMNTMwOTY3NzY1NTgzIgwVQ89ctFDKHje5VHYqwQKZ%2BjkAGweYm1FCCEmJ6sTkjmPBXKXL09764ijyjRZeuaTqKHIZje4tiH3DhcDiaVQSZGHh%2Fa872vjDIIng1ab1ZmD0ft1%2Fs8VHlz5JzRjqCCWFxYfpkPweqX7%2BoFljoaG7rqPFlm7lbk244SSR2k9qp%2BDsG4dEiCDmm%2FxiZA8%2BkezVYrv%2B3dV7uZ9omZI0fOFs0P%2BOTwRQe%2BdsxrQaHsTRMXhWlsjWfXyPj306WXekciIwjW6NcfgpqWUgKEbIU6iF0Vl31DdgtkiJ8lPbYZaH%2FaFXp8v%2F2MRYbbb2ah8chLPze2czBMYuBun7kXZHlZfLjhKPsu2fDeJeKawde9SMKNbhxCcBB88wR5yhgI7bRF3vIUPuDm0K%2Fc7P%2BnldHJfFo7y4nfc0J8Ppq9T6nq1rStU%2FWRpz%2FbNTxSBPafSiYm4wsP3StQY6swKQOviaxFGfjOVy78GuhSeUux5wrlB4HONkBXfuNJyKsgQdqNUmJFaUqbJEHKVaKfzi%2F3HIRI1wMl49BW5QZ8O%2F8eWrDKmNWS3Tq%2BWdG6C2z0h8CrHCOAjh4Zu%2ByzF77N7Nq0o88H%2BBTkvLbPXXv7iIIitnXyKpm98rkRF3xzRZaw54wnID8eS72UVcHyEdBwSQRF8bQhdFsaA3k7xdiwOohofbJeFHQPBDrSdRQC7Jbawlsjk%2F4PFhdI0%2B2eUGtpdO1rCenDM3X77JbpAFct4ntcuKOc7xNOxnD59QLqM4bUJ7JgcYDyWReayrgz%2FNKyVNSY0XO1wMoCWR1AuiotbBOPIvDRalrMHgl2nocBtZifE%2BmJF3Eztl38X5KzGeU2J8xJq4DXTXaewFoAAqT2irkIe7&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240808T125427Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAXXIBFEJHUEIGI6GN%2F20240808%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Signature=bebc6a43f9ea2be4d932e34941710bd01774114132bdd9a8256d3a96a6bcac97', engine="python")
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
