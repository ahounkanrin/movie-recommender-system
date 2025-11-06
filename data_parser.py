import os
import numpy as np
import polars as pl

from matplotlib import pyplot as plt
from tqdm import tqdm

def parse_data(data):
    user_counter = 0
    movie_counter = 0
    num_users = data["userId"].n_unique()
    num_movies = data["movieId"].n_unique()

    user_id_to_index = {}
    index_to_user_id = []
    data_by_user = [[] for i in range(num_users)]

    movie_id_to_index = {}
    index_to_movie_id = []
    data_by_movie = [[] for i in range(num_movies)]

    for i in tqdm(range(len(data))):
        user_id, movie_id, rating, _ = data[i]
        user_id, movie_id, rating = user_id.item(), movie_id.item(), rating.item()
        
        if user_id not in user_id_to_index.keys():
            user_id_to_index[user_id] = user_counter
            index_to_user_id.append(user_id)
            user_counter += 1

        if movie_id not in movie_id_to_index.keys():
            movie_id_to_index[movie_id] = movie_counter
            index_to_movie_id.append(movie_id)
            movie_counter += 1
        
        user_index = user_id_to_index[user_id]
        movie_index = movie_id_to_index[movie_id]
        
        data_by_user[user_index].append((movie_index, rating))
        data_by_movie[movie_index].append((user_index, rating))
            
    return data_by_user, data_by_movie, index_to_user_id, index_to_movie_id


def plot_rating_distribution(data_by_user, data_by_movie):
    user_degree_list = np.array([len(x) for x in data_by_user])
    movie_degree_list = np.array([len(x) for x in data_by_movie])

    user_degree, user_degree_frequency = np.unique(user_degree_list, return_counts=True)
    movie_degree, movie_degree_frequency = np.unique(movie_degree_list, return_counts=True)

    
    plt.scatter(movie_degree, movie_degree_frequency, label="movies", s=10, color="b")
    plt.scatter(user_degree, user_degree_frequency, label="users", marker='D', s=10, color="m")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("./outputs/ratings_distribution.pdf")
    
if __name__ == "__main__":

    DATA_DIR = "./data/ml-32m"
    data = pl.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    # data_subset_size = 10000000
    # data = data[:data_subset_size]

    data_by_user, data_by_movie, index_to_user_id, index_to_movie_id = parse_data(data)
    plot_rating_distribution(data_by_user, data_by_movie)
