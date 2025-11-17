import streamlit as st 
import numpy as np
import polars as pl
import random
import requests

from huggingface_hub import hf_hub_download

random.seed(42)
np.random.seed(42) 

lambda_ = 0.1
gamma_ = 0.1
tau_ = 0.1
embedding_dim = 8
num_steps = 10
I = np.eye(embedding_dim)
num_recommendations = 10
num_columns = 5

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

@st.cache_data(show_spinner=False)
def get_movie_details_from_tmdb(movie_ids, language, api_key=TMDB_API_KEY):
    base_poster_url = "https://image.tmdb.org/t/p/w500"  # w500 is the image width
    poster_urls = {}
    movie_titles = {}
    release_years = {}

    for tmdb_id in movie_ids:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        # params = {"api_key": api_key, "language": "en-US"}
        params = {"api_key": api_key, "language": language}
        response = requests.get(url, params=params)
        movie_details = response.json()
    
        poster_path = movie_details.get("poster_path", None)
        movie_title = movie_details.get("title", None)
        release_date = movie_details.get("release_date", None)

        if release_date is not None:
            release_year = release_date.split("-")[0]
        else:
            release_year = None

        movie_titles[tmdb_id] = movie_title
        release_years[tmdb_id] = release_year

        if poster_path is not None:
            poster_urls[tmdb_id] = base_poster_url + poster_path
        else:
            poster_urls[tmdb_id] = "https://dummyimage.com/200x300/cccccc/000000&text=No+Image"

    return poster_urls, movie_titles, release_years

@st.cache_resource
def load_movie_data():
    movie_data_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/movies.parquet"
    index_to_movie_id_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/index_to_movie_id.parquet"
    movie_id_to_index_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/movie_id_to_index.parquet"
    rating_counts_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/rating_counts.parquet"
    movie_id_links_url = "https://huggingface.co/datasets/ahounkanrin/ml-32m/resolve/main/links.parquet"

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
    
    return (rating_counts, index_to_movie_id,
            movie_id_to_index, movie_id_to_movie_title,
            movie_titles, movie_data, ml_id_to_tmdb_id)


st.title("Movie Recommendation App")

(rating_counts, index_to_movie_id,
            movie_id_to_index, movie_id_to_movie_title,
            movie_titles, movie_data, ml_id_to_tmdb_id) = load_movie_data()

user_biases, movie_biases, user_embeddings, movie_embeddings = load_model()

# languanges_list = get_languages_list_from_tmdb()

selected_movie = st.selectbox("Select a movie you like", movie_titles, index=None)
rating = st.slider(f"How would you rate this movie? (0 - 5 stars)", 0.0, 5.0, 2.5, step=0.5)
selected_language = st.selectbox("Select language",
                                 options=["fr-FR", "en-EN", "de-DE", "es-ES", "it-IT", "zh-CN", "ar-AR"],
                                 index=0)
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

    predicted_ratings = -np.inf * np.ones(shape=(num_movies))

    for n in range(num_movies):
        # Do not recommend the same movie and movies with too few ratings
        if n  == movie_index or rating_counts[index_to_movie_id[n]] < 100:
            continue
        
        predicted_ratings[n] = np.dot(dummy_user_embedding, movie_embeddings[n])\
                                + 0.05 * movie_biases[n]

    ranked_movies = np.argsort(predicted_ratings)[-num_recommendations:]
    topk_movies_indices = ranked_movies[::-1]

    topk_movies_ids = [index_to_movie_id[x] for x in topk_movies_indices]
    topk_movies_titles_ml = [movie_id_to_movie_title[x] for x in topk_movies_ids] # MovieLens titles

    tmdb_ids = [ml_id_to_tmdb_id[x] for x in topk_movies_ids]

    poster_urls_dict, movie_titles_dict, release_years_dict = get_movie_details_from_tmdb(tmdb_ids,
                                                                                          language=selected_language)
    poster_urls = [poster_urls_dict[x] for x in tmdb_ids]
    topk_movies_titles = [movie_titles_dict[x] for x in tmdb_ids] # TMDB titles
    topk_release_years = [release_years_dict[x] for x in tmdb_ids]

    for i in range(0, len(poster_urls), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            if i + j < len(poster_urls):
                # Display poster with TMDB titles if there exist. Else, display MovieLens title
                if topk_movies_titles[i+j] is not None:
                    col.image(poster_urls[i+j],
                            caption=f"{topk_movies_titles[i+j]} ({topk_release_years[i+j]})",
                            width="stretch")
                else:
                    col.image(poster_urls[i+j],
                            caption=f"{topk_movies_titles_ml[i+j]}",
                            width="stretch")
