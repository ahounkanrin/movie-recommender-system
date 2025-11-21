import numpy as np
import polars as pl
import os
import random


random.seed(42)
np.random.seed(42) 

lambda_ = 0.1
gamma_ = 0.1
tau_ = 2.0
embedding_dim = 8
num_steps = 10
num_recommendations = 10
I = np.eye(embedding_dim)

MODEL_DIR = "./deployed_model"
DATA_DIR = "./data/ml-32m"


def load_model():
    # model = np.load(os.path.join(MODEL_DIR, f"model.npz"))
    model = np.load(os.path.join("models", f"model_with_feature_embeding_dim_{embedding_dim}_32m.npz"))
    # user_biases = model["user_biases"]
    movie_biases = model["movie_biases"]
    # user_embeddings = model["user_embeddings"]
    movie_embeddings = model["movie_embeddings"]
    return movie_biases, movie_embeddings

def load_movie_data():
    movie_data_url = "processed_data/filtered_movies.parquet"
    index_to_movie_id_url = "processed_data/index_to_movie_id.parquet"
    movie_id_to_index_url = "processed_data/movie_id_to_index.parquet"
    rating_counts_url = "processed_data/rating_counts.parquet"

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
    
    return (rating_counts, index_to_movie_id,
            movie_id_to_index, movie_id_to_movie_title,
            movie_titles, movie_data)

movie_biases, movie_embeddings = load_model()

(rating_counts, index_to_movie_id,
movie_id_to_index, movie_id_to_movie_title,
movie_titles, movie_data) = load_movie_data()

# movie_title = "Lord of the Rings: The Fellowship of the Ring, The (2001)"
# movie_title = "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)"
# movie_title = "Love and Other Catastrophes (1996)"
# movie_title = "Blood Diamond (2006)"
# movie_title = "Gladiator (2000)"
# movie_title = "Fight For Space (2016)"
# movie_title = "Kung Fu Panda (2008)"
movie_title = "Toy Story (1995)"
# movie_title = "Aladdin (1992)"

dummy_user_rating = 5
movie_id = movie_data.filter(pl.col("title") == movie_title).select("movieId").to_series().to_list()[0]
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

num_movies = len(movie_biases)
predicted_ratings = - np.inf * np.ones(shape=(num_movies))

for n in range(num_movies):
    # Do not recommend the same movie or  movies with too few ratings
    if n  == movie_index or rating_counts[index_to_movie_id[n]] < 100:
        continue
    
    predicted_ratings[n] = np.dot(dummy_user_embedding, movie_embeddings[n])\
                            + 0.05 * movie_biases[n]

ranked_movies = np.argsort(predicted_ratings)[-num_recommendations:]
topk_movies_indices = ranked_movies[::-1]

topk_movies_ids = [index_to_movie_id[x] for x in topk_movies_indices]
topk_movies_titles = [movie_id_to_movie_title[x] for x in topk_movies_ids]

print(f"Top {num_recommendations} recommendations:")
for i in range(num_recommendations):
    print(f"\t{i+1}: {topk_movies_titles[i]}")
