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
embedding_dim = 8
num_steps = 1
I = np.eye(embedding_dim)

MODEL_DIR = "./models"
# DATA_DIR = "./data/ml-32m"
DATA_DIR = "./data/ml-25m"

with open("./data/processed/data_ml_25m.pkl", "rb") as f:
    rating_data = pickle.load(f)

data_by_user, data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index = rating_data

movie_data = pl.read_csv(os.path.join(DATA_DIR, "movies.csv"))

model = np.load(os.path.join(MODEL_DIR, f"model_embeding_dim_{embedding_dim}_25m.npz"))
user_biases = model["user_biases"]
movie_biases = model["movie_biases"]
user_embeddings = model["user_embeddings"]
movie_embeddings = model["movie_embeddings"]

# movie_title = "Lord of the Rings: The Fellowship of the Ring, The (2001)"# movie_title = "Kung Fu Panda (2008)"
movie_title = "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)"
# movie_title = "Countdown to Zero (2010)"
# movie_title = "Love and Other Catastrophes (1996)"
# movie_title = "Blood Diamond (2006)"
# movie_title = "Gladiator (2000)"
# movie_title = "Fight For Space (2016)"

dummy_user_rating = 5
movie_id = movie_data.filter(pl.col("title") == movie_title).select("movieId").item()
movie_index = movie_id_to_index[movie_id]

dummy_user_bias = 0. 
dummy_user_embedding = np.random.normal(loc=0, scale=np.sqrt(1/embedding_dim), size=(embedding_dim))

for step in range(num_steps):
    bias = lambda_ * (dummy_user_rating - movie_biases[movie_index]\
                        - np.dot(dummy_user_embedding, movie_embeddings[movie_index]))
    bias /= (lambda_ + gamma_)
    dummy_user_bias = bias

    A = np.outer(movie_embeddings[movie_index], movie_embeddings[movie_index])
    b = lambda_ * (dummy_user_rating - dummy_user_bias - movie_biases[movie_index]) * movie_embeddings[movie_index]
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
                            + 0.05 * movie_biases[n]
ranked_movies = np.flip(np.argsort(predicted_ratings))
topk_movies_indices = ranked_movies[:k]

topk_movies_ids = [index_to_movie_id[x] for x in topk_movies_indices]
topk_movies_titles = [movie_data.filter(pl.col("movieId") == x).select("title").item()\
                        for x in topk_movies_ids]

print(f"Top {k} recommendations:")
for i in range(k):
    print(f"\t{i+1}: {topk_movies_titles[i]}")