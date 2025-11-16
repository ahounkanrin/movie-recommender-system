import streamlit as st 
import numpy as np
import polars as pl
import os
import random
import requests
import re 

from huggingface_hub import hf_hub_download
from tqdm import tqdm

random.seed(42)
np.random.seed(42) 

lambda_ = 0.1
gamma_ = 0.1
tau_ = 0.1
embedding_dim = 8
num_steps = 10
I = np.eye(embedding_dim)
num_recommendations = 10

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]


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

@st.cache_data(show_spinner=True)
def get_poster_url(movie_ids, api_key=TMDB_API_KEY):
    base_poster_url = "https://image.tmdb.org/t/p/w500"  # w500 is the image width
    poster_urls = {}

    for tmdb_id in movie_ids:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {"api_key": api_key, "language": "en-US"}
        response = requests.get(url, params=params)
        movie_details = response.json()
    
        post_path = movie_details.get("poster_path", None)
        if post_path is not None:
            poster_urls[tmdb_id] = base_poster_url + post_path
        else:
            poster_urls[tmdb_id] = "https://dummyimage.com/200x300/cccccc/000000&text=No+Image"

    return poster_urls

@st.cache_data
def save_poster_url(movie_ids, api_key=TMDB_API_KEY):
    base_poster_url = "https://image.tmdb.org/t/p/w500" 
    poster_urls = {}

    for tmdb_id in tqdm(movie_ids):
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {"api_key": api_key, "language": "en-US"}
        response = requests.get(url, params=params)
        movie_details = response.json()
        post_path = movie_details.get("poster_path", None)
        if post_path is not None:
            poster_urls[tmdb_id] = base_poster_url + post_path
        else:
            poster_urls[tmdb_id] = "https://dummyimage.com/200x300/cccccc/000000&text=No+Image"

    poster_urls_df = pl.DataFrame({
        "movieId": poster_urls.keys(),
        "url": poster_urls.values()
    })
    poster_urls_df.write_parquet("./data/processed/tmdb_poster_urls.parquet")
    return poster_urls


@st.cache_resource
def load_movie_data():
    movie_data_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/movies.parquet"
    index_to_movie_id_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/index_to_movie_id.parquet"
    movie_id_to_index_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/movie_id_to_index.parquet"
    rating_counts_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/rating_counts.parquet"
    movie_id_links_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/links.parquet"
    # tmdb_posters_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/tmdb_poster_urls.parquet"

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
    
    movie_id_links_df = pl.read_parquet(movie_id_links_url)
    ml_id_to_tmdb_id = dict(zip(movie_id_links_df["movieId"].to_list(),
                                movie_id_links_df["tmdbId"].to_list() 
                                ))
    
    # tmdb_posters_df = pl.read_parquet(tmdb_posters_url)
    # tmdb_posters_dict = dict(zip(tmdb_posters_df["movieId"].to_list(),
    #                              tmdb_posters_df["url"].to_list()

    # ))
    
    # return (rating_counts, index_to_movie_id,
    #         movie_id_to_index, movie_id_to_movie_title,
    #         movie_titles, movie_data, ml_id_to_tmdb_id, tmdb_posters_dict)
    return (rating_counts, index_to_movie_id,
            movie_id_to_index, movie_id_to_movie_title,
            movie_titles, movie_data, ml_id_to_tmdb_id)


st.title("Movie Recommendation App")

# (rating_counts, index_to_movie_id,
# movie_id_to_index, movie_id_to_movie_title,
# movie_titles, movie_data, ml_id_to_tmdb_id, tmdb_posters_dict) = load_movie_data()
(rating_counts, index_to_movie_id,
            movie_id_to_index, movie_id_to_movie_title,
            movie_titles, movie_data, ml_id_to_tmdb_id) = load_movie_data()

# tmdb_movies_ids = [ml_id_to_tmdb_id[x] for x in index_to_movie_id]
# save_poster_url(tmdb_movies_ids)
# st.write("URLs SAVED")

user_biases, movie_biases, user_embeddings, movie_embeddings = load_model()

selected_movie = st.selectbox("Select a movie you like", movie_titles, index=None)
rating = st.slider(f"How would you rate this movie? (0 - 5 stars)", 0.0, 5.0, 2.5, step=0.5)
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

    tmdb_ids = [ml_id_to_tmdb_id[x] for x in topk_movies_ids]
    poster_urls_dict = get_poster_url(tmdb_ids)
    poster_urls = [poster_urls_dict[x] for x in tmdb_ids]
    # poster_urls = [tmdb_posters_dict[x] for x in tmdb_ids]

    num_columns = 5
    for i in range(0, len(poster_urls), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            if i + j < len(poster_urls):
                col.image(poster_urls[i+j], caption=topk_movies_titles[i+j], width="stretch")
                