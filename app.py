import streamlit as st 
import numpy as np
import polars as pl
import os
import random

from huggingface_hub import hf_hub_download
from data_parser import parse_data

random.seed(42)
np.random.seed(42) 

lambda_ = 0.1
gamma_ = 0.1
tau_ = 0.1
embedding_dim = 8
num_steps = 10
I = np.eye(embedding_dim)
num_recommendations = 10


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
    repo_id="ahounkanrin/ml-32m",
    filename=f"model_embeding_dim_{embedding_dim}_32m.npz",
    repo_type="dataset")
    model = np.load(model_path)
    user_biases = model["user_biases"]
    movie_biases = model["movie_biases"]
    user_embeddings = model["user_embeddings"]
    movie_embeddings = model["movie_embeddings"]

    return user_biases, movie_biases, user_embeddings, movie_embeddings


@st.cache_resource
def load_movie_data():
    movie_data_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/movies.parquet"
    index_to_movie_id_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/index_to_movie_id.parquet"
    movie_id_to_index_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/movie_id_to_index.parquet"
    rating_counts_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/rating_counts.parquet"

    movie_data = pl.read_parquet(movie_data_url)
    movie_id_to_movie_title = {row["movieId"]: row["title"] for row in movie_data.iter_rows(named=True)}
    movie_titles = sorted(movie_id_to_movie_title.values())

    index_to_movie_id_df = pl.read_parquet(index_to_movie_id_url)
    index_to_movie_id = index_to_movie_id_df["movieId"].to_list()

    movie_id_to_index_df = pl.read_parquet(movie_id_to_index_url)
    movie_id_to_index = dict(zip(movie_id_to_index_df["movieId"], movie_id_to_index_df["index"]))

    rating_counts_df = pl.read_parquet(rating_counts_url)
    rating_counts = dict(zip(rating_counts_df["movieId"].to_list(),
                             rating_counts_df["count"].to_list()))

    return rating_counts, index_to_movie_id, movie_id_to_index, movie_id_to_movie_title, movie_titles, movie_data


st.title("Movie Recommendation App")

rating_counts, index_to_movie_id, movie_id_to_index, movie_id_to_movie_title, movie_titles, movie_data = load_movie_data()
user_biases, movie_biases, user_embeddings, movie_embeddings = load_model()
# data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index = load_rating_data()

selected_movie = st.selectbox("Select a movie", movie_titles, index=None)
rating = st.slider(f"Rate this movie", 0.0, 5.0, 2.5, step=0.5)
recommendation_request = st.button("Show recommendations")

num_movies = len(movie_biases)
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
        if rating_counts[index_to_movie_id[n]] < 100:
            continue
        
        predicted_ratings[n] = np.dot(dummy_user_embedding, movie_embeddings[n])\
                                + 0.05 * movie_biases[n]

    ranked_movies = np.flip(np.argsort(predicted_ratings))
    topk_movies_indices = ranked_movies[:num_recommendations]

    topk_movies_ids = [index_to_movie_id[x] for x in topk_movies_indices]
    topk_movies_titles = [movie_id_to_movie_title[x] for x in topk_movies_ids]

    for i in range(num_recommendations):
        st.write(f"\t{i+1}. {topk_movies_titles[i]}")