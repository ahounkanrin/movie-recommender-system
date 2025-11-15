import numpy as np
import polars as pl
import os
import random

from matplotlib import pyplot as plt
from numba import jit, prange

random.seed(42)
np.random.seed(42) 

lambda_ = 0.1
gamma_ = 0.1
tau_ = 0.1
num_epochs = 20
embedding_dim = 2

I = np.eye(embedding_dim)

@jit(nopython=True, parallel=True)
def train(data_by_user_user_index_offsets_train,
            data_by_user_movie_indices_train,
            data_by_user_ratings_train,
            data_by_movie_movie_index_offsets_train,
            data_by_movie_user_indices_train,
            data_by_movie_ratings_train,
            num_users,
            num_movies,
            user_biases,
            movie_biases,
            user_embeddings,
            movie_embeddings
            ):
    
    train_losses = np.zeros(shape=(num_epochs)) 
    train_errors = np.zeros(shape=(num_epochs))

    for epoch in range(num_epochs):
        
        for m in prange(num_users):
            # update user biases
            bias = 0
            movie_counter = 0
            start_idx = data_by_user_user_index_offsets_train[m]
            end_idx = data_by_user_user_index_offsets_train[m+1]
            for i in range(start_idx, end_idx):
                n = data_by_user_movie_indices_train[i]
                r = data_by_user_ratings_train[i]
                bias += lambda_ * (r - movie_biases[n]\
                                   - np.dot(user_embeddings[m], movie_embeddings[n]))
                movie_counter += 1
            bias /= (lambda_ * movie_counter + gamma_)
            user_biases[m] = bias

            # update user vectors
            A = np.zeros(shape=(embedding_dim, embedding_dim))
            b = np.zeros(shape=(embedding_dim))

            for i in range(start_idx, end_idx):
                n = data_by_user_movie_indices_train[i]
                r = data_by_user_ratings_train[i]

                A += np.outer(movie_embeddings[n], movie_embeddings[n])
                b += lambda_ * (r - user_biases[m] - movie_biases[n]) * movie_embeddings[n]
            
            A = lambda_ * A + tau_ * I
            user_embeddings[m] = np.linalg.solve(A, b)

        for n in prange(num_movies):
            # update movie biases
            bias = 0
            user_counter = 0
            start_idx = data_by_movie_movie_index_offsets_train[n]
            end_idx = data_by_movie_movie_index_offsets_train[n+1]
            for i in range(start_idx, end_idx):
                    m = data_by_movie_user_indices_train[i]
                    r = data_by_movie_ratings_train[i]
                    bias += lambda_ * (r - user_biases[m]\
                                       - np.dot(user_embeddings[m], movie_embeddings[n]))
                    user_counter += 1
            bias /= (lambda_ * user_counter + gamma_)
            movie_biases[n] = bias

            # update movie vectors
            A = np.zeros(shape=(embedding_dim, embedding_dim))
            b = np.zeros(shape=(embedding_dim))

            for i in range(start_idx, end_idx):
                m = data_by_movie_user_indices_train[i]
                r = data_by_movie_ratings_train[i]

                A += np.outer(user_embeddings[m], user_embeddings[m])
                b += lambda_ * (r - user_biases[m] - movie_biases[n]) * user_embeddings[m]
            
            A = lambda_ * A + tau_ * I
            
            movie_embeddings[n] = np.linalg.solve(A, b)

        # compute train loss
        loss_train = 0
        rmse_train = 0
        rating_counter_train = 0
        for m in prange(num_users):
            start_idx = data_by_user_user_index_offsets_train[m]
            end_idx = data_by_user_user_index_offsets_train[m+1]

            for i in range(start_idx, end_idx):
                n = data_by_user_movie_indices_train[i]
                r = data_by_user_ratings_train[i]

                loss_train += (lambda_/2) * (r - user_biases[m] - movie_biases[n]
                                             - np.dot(user_embeddings[m], movie_embeddings[n]))**2
                rmse_train += (r - user_biases[m] - movie_biases[n]
                               - np.dot(user_embeddings[m], movie_embeddings[n]))**2
                rating_counter_train += 1
        
        loss_train += (gamma_/2) * (np.sum(user_biases**2) + np.sum(movie_biases**2))
        loss_train += (tau_/2) * (np.linalg.norm(user_embeddings)**2 + np.linalg.norm(movie_embeddings)**2)
        rmse_train = np.sqrt(rmse_train/rating_counter_train)
        
        train_losses[epoch] = loss_train
        train_errors[epoch] = rmse_train

        print("Epoch:", epoch+1, "\t train_loss = ", loss_train, "\t rmse_train = ",rmse_train)
    
    model = (user_biases, user_embeddings, movie_biases, movie_embeddings)
    return train_losses,train_errors, model


def plot_errors_and_losses(train_losses, train_errors):    
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(1, len(train_losses) + 1),train_losses, label="Train", color="b")
    ax.legend()
    fig.suptitle("Negative log likelihood")
    ax.grid(True)
    ax.set_xlabel("Epoch")
    plt.savefig(f"./outputs/plots/bias_and_embedding_model_nll_32M_full_dataset_embed_{embedding_dim}.pdf")
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(1, len(train_errors) + 1), train_errors, label="Train", color="b")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    # plt.suptitle("Mean squared error")
    ax.grid(True)
    plt.savefig(f"./outputs/plots/bias_and_embeddding_model_rmse_32M_full_dataset_embed_{embedding_dim}.pdf")
    plt.close()

if __name__ == "__main__":

    DATA_DIR = "./data/processed"
    data = np.load(os.path.join(DATA_DIR, "flat_data_32m_train_full.npz"))

    data_by_user_user_index_offsets_train = data["data_by_user_user_index_offsets_train"]
    data_by_user_movie_indices_train = data["data_by_user_movie_indices_train"]
    data_by_user_ratings_train = data["data_by_user_ratings_train"]
    data_by_movie_movie_index_offsets_train = data["data_by_movie_movie_index_offsets_train"]
    data_by_movie_user_indices_train = data["data_by_movie_user_indices_train"]
    data_by_movie_ratings_train = data["data_by_movie_ratings_train"]
    num_users = int(data["num_users"].item())
    num_movies = int(data["num_movies"].item())
    
    user_biases = np.zeros(shape=(num_users))
    movie_biases = np.zeros(shape=(num_movies))

    user_embeddings = np.random.normal(loc=0, scale=np.sqrt(1/embedding_dim), size=(num_users, embedding_dim))
    movie_embeddings = np.random.normal(loc=0, scale=np.sqrt(1/embedding_dim), size=(num_movies, embedding_dim))
 
    train_losses, train_errors, model =  train(data_by_user_user_index_offsets_train,
                                                data_by_user_movie_indices_train,
                                                data_by_user_ratings_train,
                                                data_by_movie_movie_index_offsets_train,
                                                data_by_movie_user_indices_train,
                                                data_by_movie_ratings_train,
                                                num_users,
                                                num_movies,
                                                user_biases,
                                                movie_biases,
                                                user_embeddings,
                                                movie_embeddings
                                                )
    
    user_biases, user_embeddings, movie_biases, movie_embeddings = model

    np.savez(f"./models/model_embeding_dim_{embedding_dim}_32m.npz",
             user_biases=user_biases, user_embeddings=user_embeddings,
             movie_biases=movie_biases, movie_embeddings=movie_embeddings)

    plot_errors_and_losses(train_losses, train_errors)
