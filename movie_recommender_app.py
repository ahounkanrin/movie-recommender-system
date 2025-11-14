import streamlit as st 
import numpy as np
import polars as pl
import os
import random
import pickle


random.seed(42)
np.random.seed(42) 

lambda_ = 0.1
gamma_ = 0.1
tau_ = 0.1
embedding_dim = 8
num_steps = 10
I = np.eye(embedding_dim)
num_recommendations = 10

MODEL_DIR = "./models"
DATA_DIR = "./data/ml-32m"


@st.cache_resource
def load_model():
    model = np.load(os.path.join(MODEL_DIR, f"model_embeding_dim_{embedding_dim}_32m.npz"))
    user_biases = model["user_biases"]
    movie_biases = model["movie_biases"]
    user_embeddings = model["user_embeddings"]
    movie_embeddings = model["movie_embeddings"]
    return user_biases, movie_biases, user_embeddings, movie_embeddings


@st.cache_resource
def load_rating_data():
    with open("./data/processed/data_ml_32m.pkl", "rb") as f:
        rating_data = pickle.load(f)

    _, data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index = rating_data
    return data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index


@st.cache_resource
def process_movie_data():
    movie_data = pl.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    movie_id_to_movie_title = {row["movieId"]: row["title"] for row in movie_data.iter_rows(named=True)}
    movie_titles = sorted(movie_id_to_movie_title.values())
    return movie_id_to_movie_title, movie_titles, movie_data


user_biases, movie_biases, user_embeddings, movie_embeddings = load_model()
data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index = load_rating_data()
movie_id_to_movie_title, movie_titles, movie_data = process_movie_data()

num_movies = len(movie_biases)

st.title("Movie Recommendation App")
selected_movie = st.selectbox("Select a movie", movie_titles, index=None)
rating = st.slider(f"Rate this movie", 0.0, 5.0, 2.5, step=0.5)
recommendation_request = st.button("Show recommendations")

if recommendation_request and selected_movie is not None:
    movie_id = movie_data.filter(pl.col("title") == selected_movie).select("movieId").to_series().to_list()[0]
    movie_index = movie_id_to_index[movie_id]

    dummy_user_bias = 0. 
    dummy_user_embedding = np.random.normal(loc=0, scale=np.sqrt(1/embedding_dim), size=(embedding_dim))

    for step in range(num_steps):
        bias = lambda_ * (rating - movie_biases[movie_index]\
                            - np.dot(dummy_user_embedding, movie_embeddings[movie_index]))
        bias /= (lambda_ + gamma_)
        dummy_user_bias = bias

        A = np.outer(movie_embeddings[movie_index], movie_embeddings[movie_index])
        b = lambda_ * (rating - dummy_user_bias - movie_biases[movie_index]) * movie_embeddings[movie_index]
        A = lambda_ * A + tau_ * I 
        dummy_user_embedding = np.linalg.solve(A, b)

    predicted_ratings = np.zeros(shape=(num_movies))

    for n in range(num_movies):
        # Do not recommend the same movie
        if n  == movie_index:
            continue

        # Filter out movies with too few ratings in the training set
        if len(data_by_movie[n]) < 100:
            continue
        
        predicted_ratings[n] = np.dot(dummy_user_embedding, movie_embeddings[n])\
                                + 0.05 * movie_biases[n]

    ranked_movies = np.flip(np.argsort(predicted_ratings))
    topk_movies_indices = ranked_movies[:num_recommendations]

    topk_movies_ids = [index_to_movie_id[x] for x in topk_movies_indices]
    topk_movies_titles = [movie_id_to_movie_title[x] for x in topk_movies_ids]

    for i in range(num_recommendations):
        st.write(f"\t{i+1}: {topk_movies_titles[i]}")