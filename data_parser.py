import os
import numpy as np
import polars as pl
import random
import pickle

from matplotlib import pyplot as plt
from tqdm import tqdm


random.seed(42)
np.random.seed(42)

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
    movie_indices = []
    ratings = []
    
    offset = 0
    for i in tqdm(range(len(data)), desc="Flatten user data"):
        for j in range(len(data[i])):
            movie_indices.append(data[i][j][0])
            ratings.append(data[i][j][1])

        offset += len(data[i])
        user_index_offsets.append(offset)

    return np.array(user_index_offsets), np.array(movie_indices), np.array(ratings)

def flatten_movie_data(data):
    movie_index_offsets = [0]
    user_indices = []
    ratings = []
    
    offset = 0
    for i in tqdm(range(len(data)), desc="Flatten movie data"):
        for j in range(len(data[i])):
            user_indices.append(data[i][j][0])
            ratings.append(data[i][j][1])
        
        offset += len(data[i])
        movie_index_offsets.append(offset)

    return np.array(movie_index_offsets), np.array(user_indices), np.array(ratings)

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


def parse_movie_features(index_to_movie_id, movie_data):
    feature_name_to_index = {}
    index_to_feature_name = []
    movie_feature_data = [[] for _ in range(len(index_to_movie_id))]

    feature_counter = 0
    for n in tqdm(range(len(index_to_movie_id)), desc="Parsing features"):
        features = movie_data.filter(pl.col("movieId") == index_to_movie_id[n]).select("genres").item()
        if features == "(no genres listed)":
            continue
        features = features.split("|")
        for genre in features:
            if genre not in feature_name_to_index.keys():
                feature_name_to_index[genre] = feature_counter
                index_to_feature_name.append(genre)
                feature_counter += 1

            movie_feature_data[n].append(feature_name_to_index[genre])
        # print(features)
        
    # print(feature_name_to_index)
    # print(movie_feature_data)
    return movie_feature_data, feature_name_to_index, index_to_feature_name


def flatten_movie_feature_data(movie_feature_data):
    movie_feature_index_offsets = [0]
    feature_indices = []
    
    offset = 0
    for i in tqdm(range(len(movie_feature_data)), desc="Flatten movie feature data"):
        for j in range(len(movie_feature_data[i])):
            feature_indices.append(movie_feature_data[i][j])
        
        offset += len(data[i])
        movie_feature_index_offsets.append(offset)

    return np.array(movie_feature_index_offsets), np.array(feature_indices)


if __name__ == "__main__":
    DATA_DIR = "./data/ml-32m"
    # DATA_DIR = "./data/ml-25m"
    data = pl.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    data = data.sort("timestamp")

    movie_data = pl.read_csv(os.path.join(DATA_DIR, "movies.csv"))

    filtered_movie_data = movie_data.filter(pl.col("movieId").is_in(data["movieId"].unique().implode()))
    filtered_movie_data.write_parquet("./data/processed/filtered_movies.parquet")
    
    data_by_user, data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index = parse_data(data)

    
    movie_feature_data, feature_name_to_index, index_to_feature_name = parse_movie_features(index_to_movie_id, movie_data)
    movie_feature_index_offsets, feature_indices = flatten_movie_feature_data(movie_feature_data)
    num_features = len(index_to_feature_name)

    np.savez("data/processed/flat_feature_data_32m.npz",
             movie_feature_index_offsets=movie_feature_index_offsets,
             feature_indices=feature_indices,
             num_features=num_features
             )

    # data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test = random_split(data_by_user, data_by_movie)

    index_to_movie_id_df = pl.DataFrame({
        "index": list((range(len(index_to_movie_id)))),
        "movieId": index_to_movie_id
    })

    movie_id_to_index_df = pl.DataFrame({
        "movieId": list(movie_id_to_index.keys()),
        "index": list(movie_id_to_index.values())
    })

    
    index_to_movie_id_df.write_parquet("./data/processed/index_to_movie_id.parquet")
    movie_id_to_index_df.write_parquet("./data/processed/movie_id_to_index.parquet")


    rating_counts_df = data.group_by("movieId").len(name="count")
    rating_counts_df.write_parquet("./data/processed/rating_counts.parquet")

    num_users = len(data_by_user)
    num_movies = len(data_by_movie)

    # plot_rating_distribution(data_by_user_train, data_by_movie_train)
    # plot_rating_distribution(data_by_user_test, data_by_movie_test)

    processed_data = data_by_user, data_by_movie, index_to_user_id, index_to_movie_id, user_id_to_index, movie_id_to_index

    with open("./data/processed/data_ml_32m.pkl", "wb") as f:
        pickle.dump(processed_data, f)

    data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test = chrono_split(data_by_user, data_by_movie)

    data_by_user_user_index_offsets_train, data_by_user_movie_indices_train, data_by_user_ratings_train = flatten_user_data(data_by_user_train)
    data_by_user_user_index_offsets_test, data_by_user_movie_indices_test, data_by_user_ratings_test = flatten_user_data(data_by_user_test)
    data_by_movie_movie_index_offsets_train, data_by_movie_user_indices_train, data_by_movie_ratings_train = flatten_movie_data(data_by_movie_train)

    np.savez("./data/processed/flat_data_32m_train.npz",
             data_by_user_user_index_offsets_train=data_by_user_user_index_offsets_train,
             data_by_user_movie_indices_train=data_by_user_movie_indices_train,
             data_by_user_ratings_train=data_by_user_ratings_train,
             data_by_user_user_index_offsets_test=data_by_user_user_index_offsets_test,
             data_by_user_movie_indices_test=data_by_user_movie_indices_test,
             data_by_user_ratings_test=data_by_user_ratings_test,
             data_by_movie_movie_index_offsets_train=data_by_movie_movie_index_offsets_train,
             data_by_movie_user_indices_train=data_by_movie_user_indices_train,
             data_by_movie_ratings_train=data_by_movie_ratings_train,
             num_users=num_users,
             num_movies=num_movies
             )
    
    data_by_user_user_index_offsets_train, data_by_user_movie_indices_train, data_by_user_ratings_train = flatten_user_data(data_by_user)
    data_by_movie_movie_index_offsets_train, data_by_movie_user_indices_train, data_by_movie_ratings_train = flatten_movie_data(data_by_movie)

    np.savez("./data/processed/flat_data_32m_train_full.npz",
             data_by_user_user_index_offsets_train=data_by_user_user_index_offsets_train,
             data_by_user_movie_indices_train=data_by_user_movie_indices_train,
             data_by_user_ratings_train=data_by_user_ratings_train,
             data_by_movie_movie_index_offsets_train=data_by_movie_movie_index_offsets_train,
             data_by_movie_user_indices_train=data_by_movie_user_indices_train,
             data_by_movie_ratings_train=data_by_movie_ratings_train,
             num_users=num_users,
             num_movies=num_movies
             )
