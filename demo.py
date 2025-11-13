import streamlit as st 
import numpy as np
import polars as pl
import os
import random
import pickle

from data_parser import parse_data, random_split, chrono_split
from tqdm import tqdm
from matplotlib import pyplot as plt


random.seed(42)
np.random.seed(42) 

lambda_ = 0.1
gamma_ = 0.1
tau_ = 0.1
embedding_dim = 2
num_steps = 1
I = np.eye(embedding_dim)

MODEL_DIR = "./models"
# DATA_DIR = "./data/ml-32m"
DATA_DIR = "./data/ml-25m"

model = np.load(os.path.join(MODEL_DIR, f"model_embeding_dim_{embedding_dim}_25m.npz"))
user_biases = model["user_biases"]
movie_biases = model["movie_biases"]
user_embeddings = model["user_embeddings"]
movie_embeddings = model["movie_embeddings"]


st.title("Movie Recommender App")
# st.write("Loading data...")

with open("./data/processed/data_ml_25m.pkl", "rb") as f:
    rating_data = pickle.load(f)

data_by_user, data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index = rating_data

movie_data = pl.read_csv(os.path.join(DATA_DIR, "movies.csv"))
movies_titles = sorted(movie_data["title"].unique().to_list())


selected_movie = st.selectbox("Select a movie you watched", movies_titles, index=None)
if selected_movie is not None:
    rating = st.slider(f"How would you rate this movie? ", 0.0, 5.0, 2.5, step=0.5)

    if st.button("Find movies you might also like"):
        movie_id = movie_data.filter(pl.col("title") == selected_movie).select("movieId").item()
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

        k = 10
        num_movies = len(movie_biases)
        predicted_ratings = np.zeros(shape=(num_movies))

        for n in range(num_movies):
            # Do not recommend the same movie
            if n  == movie_index:
                continue

            # Filter out movies with too few ratings in the training set
            if len(data_by_movie[n]) < 100:
                continue
            
            predicted_ratings[n] = np.dot(dummy_user_embedding, movie_embeddings[n])\
                                    + 0.0 * movie_biases[n]
        ranked_movies = np.flip(np.argsort(predicted_ratings))
        topk_movies_indices = ranked_movies[:k]

        topk_movies_ids = [index_to_movie_id[x] for x in topk_movies_indices]
        topk_movies_titles = [movie_data.filter(pl.col("movieId") == x).select("title").item()\
                                for x in topk_movies_ids]

        # st.write(f"You might also like:")
        for i in range(k):
            st.write(f"\t{i+1}: {topk_movies_titles[i]}")