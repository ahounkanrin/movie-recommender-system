import os
import numpy as np
import polars as pl
import random
import pickle

from matplotlib import pyplot as plt
from tqdm import tqdm


random.seed(42)

def parse_data(data):
    num_users = data["userId"].n_unique()
    num_movies = data["movieId"].n_unique()
    
    user_id_to_index = {}
    index_to_user_id = []
    data_by_user = [[] for i in range(num_users)]

    movie_id_to_index = {}
    index_to_movie_id = []
    data_by_movie = [[] for i in range(num_movies)]

    user_counter = 0
    movie_counter = 0
    for user_id, movie_id, rating, _ in tqdm(data.iter_rows(), desc="Data parsing"):

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
            
    return (data_by_user, data_by_movie, index_to_user_id, index_to_movie_id,
            user_id_to_index, movie_id_to_index)


def random_split(data_by_user, data_by_movie):
    data_by_user_train = [[] for i in range(len(data_by_user))]
    data_by_user_test = [[] for i in range(len(data_by_user))]
    data_by_movie_train = [[] for i in range(len(data_by_movie))]
    data_by_movie_test = [[] for i in range(len(data_by_movie))]

    for i in tqdm(range(len(data_by_user)), desc="Train-test random split"):
        for j in range(len(data_by_user[i])):
            movie_index = data_by_user[i][j][0]
            rating = data_by_user[i][j][1]

            u = random.uniform(0, 1)
            if u <= 0.9:
                data_by_user_train[i].append(data_by_user[i][j])
                data_by_movie_train[movie_index].append((i, rating))
            else:
                data_by_user_test[i].append(data_by_user[i][j])
                data_by_movie_test[movie_index].append((i, rating))      
    
    return data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test


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
    plt.savefig("./outputs/plots/test_ratings_distribution_small.pdf")


def flatten_user_data(data):
    user_index_offsets = [0]
    movie_indexes = []
    user_indexes = []
    ratings = []
    
    offset = 0
    for i in tqdm(range(len(data)), desc="Flatten user data"):
        for j in range(len(data[i])):
            user_indexes.append(i)
            movie_indexes.append(data[i][j][0])
            ratings.append(data[i][j][1])

        offset += len(data[i])
        user_index_offsets.append(offset)

    return np.array(user_index_offsets), np.array(movie_indexes), np.array(ratings)

def flatten_movie_data(data):
    movie_index_offsets = [0]
    movie_indexes = []
    user_indexes = []
    ratings = []
    
    offset = 0
    for i in tqdm(range(len(data)), desc="Flatten movie data"):
        for j in range(len(data[i])):
            movie_indexes.append(i)
            user_indexes.append(data[i][j][0])
            ratings.append(data[i][j][1])
        
        offset += len(data[i])
        movie_index_offsets.append(offset)

    return np.array(movie_index_offsets), np.array(user_indexes), np.array(ratings)

def chrono_split(data_by_user, data_by_movie):
    data_by_user_train = [[] for i in range(len(data_by_user))]
    data_by_user_test = [[] for i in range(len(data_by_user))]
    data_by_movie_train = [[] for i in range(len(data_by_movie))]
    data_by_movie_test = [[] for i in range(len(data_by_movie))]

    for i in tqdm(range(len(data_by_user)), desc="Train-test chrono split"):
        for j in range(len(data_by_user[i])):
            movie_index = data_by_user[i][j][0]
            rating = data_by_user[i][j][1]

            if j < int(0.9 * len(data_by_user[i])):
                data_by_user_train[i].append(data_by_user[i][j])
                data_by_movie_train[movie_index].append((i, rating))
            else:
                data_by_user_test[i].append(data_by_user[i][j])
                data_by_movie_test[movie_index].append((i, rating))      
    
    return data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test

if __name__ == "__main__":
    DATA_DIR = "./data/ml-32m"
    data = pl.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    # data = data.sort("timestamp")
    
    data_by_user, data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index = parse_data(data)
    # data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test = random_split(data_by_user, data_by_movie)

    # plot_rating_distribution(data_by_user_train, data_by_movie_train)
    # plot_rating_distribution(data_by_user_test, data_by_movie_test)
    processed_data = data_by_user, data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index

    with open("./data/processed/data_ml_32m.pkl", "wb") as f:
        pickle.dump(processed_data, f)