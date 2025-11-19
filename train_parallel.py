import numpy as np
import polars as pl
import os
import random
import argparse

from matplotlib import pyplot as plt
from numba import jit, prange

random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_", type=float, default=0.1)
parser.add_argument("--gamma_", type=float, default=0.1)
parser.add_argument("--tau_", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--embedding_dim", type=int, default=8)
args = parser.parse_args()

# Extract Argparse arguments for Numba
lambda_ = args.lambda_
gamma_ = args.gamma_
tau_ = args.tau_
num_epochs = args.num_epochs
embedding_dim = args.embedding_dim

I = np.eye(embedding_dim)

@jit(nopython=True, parallel=True)
def train(data_by_user_user_index_offsets_train, 
            data_by_user_movie_indices_train,
            data_by_user_ratings_train,
            data_by_user_user_index_offsets_test,
            data_by_user_movie_indices_test,
            data_by_user_ratings_test,
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
    test_losses = np.zeros(shape=(num_epochs))
    train_errors = np.zeros(shape=(num_epochs))
    test_errors = np.zeros(shape=(num_epochs))

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

        # compute test loss
        loss_test = 0
        rating_counter_test = 0
        rmse_test = 0
        for m in prange(num_users):
            start_idx = data_by_user_user_index_offsets_test[m]
            end_idx = data_by_user_user_index_offsets_test[m+1]

            for i in range(start_idx, end_idx):
                n = data_by_user_movie_indices_test[i]
                r = data_by_user_ratings_test[i]

                loss_test += (lambda_/2) * (r - user_biases[m] - movie_biases[n]
                                            - np.dot(user_embeddings[m], movie_embeddings[n]))**2
                rmse_test += (r - user_biases[m] - movie_biases[n]
                              - np.dot(user_embeddings[m], movie_embeddings[n]))**2
                rating_counter_test += 1
        
        loss_test += (gamma_/2) * (np.sum(user_biases**2) + np.sum(movie_biases**2))
        loss_test += (tau_/2) * (np.linalg.norm(user_embeddings)**2 + np.linalg.norm(movie_embeddings)**2)
        rmse_test = np.sqrt(rmse_test/rating_counter_test)

        
        train_losses[epoch] = loss_train
        test_losses[epoch] = loss_test
        train_errors[epoch] = rmse_train
        test_errors[epoch] = rmse_test

        print("Epoch:", epoch+1, "\t train_loss = ", loss_train, "\t test_loss = ", loss_test,\
            "\t rmse_train = ",rmse_train, "\t rmse_test = ", rmse_test)
    
    model = (user_biases, user_embeddings, movie_biases, movie_embeddings)
    return train_losses, test_losses, train_errors, test_errors, model


def plot_errors_and_losses(train_losses, test_losses, train_errors, test_errors):    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.arange(1, len(train_losses) + 1),train_losses, label="Train", color="b")
    ax[1].plot(np.arange(1, len(test_losses) + 1), test_losses, label="Test", color="r", linestyle="--")
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Negative log likelihood")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[1].set_xlabel("Epoch")
    plt.savefig(f"./outputs/plots/bias_and_embedding_model_nll_32m_embed{embedding_dim}.pdf")
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(1, len(train_errors) + 1), train_errors, label="Train", color="b")
    ax.plot(np.arange(1, len(test_errors) + 1), test_errors, label="Test", color="r", linestyle="--")
    ax.legend()
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.grid(True)
    plt.savefig(f"./outputs/plots/bias_and_embeddding_model_rmse_32m_embed{embedding_dim}.pdf")
    plt.close()

if __name__ == "__main__":

    DATA_DIR = "./data/processed"
    data = np.load(os.path.join(DATA_DIR, "flat_data_32m_train.npz"))

    data_by_user_user_index_offsets_train = data["data_by_user_user_index_offsets_train"]
    data_by_user_movie_indices_train = data["data_by_user_movie_indices_train"]
    data_by_user_ratings_train = data["data_by_user_ratings_train"]
    data_by_user_user_index_offsets_test = data["data_by_user_user_index_offsets_test"]
    data_by_user_movie_indices_test = data["data_by_user_movie_indices_test"]
    data_by_user_ratings_test = data["data_by_user_ratings_test"]
    data_by_movie_movie_index_offsets_train = data["data_by_movie_movie_index_offsets_train"]
    data_by_movie_user_indices_train = data["data_by_movie_user_indices_train"]
    data_by_movie_ratings_train = data["data_by_movie_ratings_train"]
    num_users = int(data["num_users"].item())
    num_movies = int(data["num_movies"].item())


    user_biases = np.zeros(shape=(num_users))
    movie_biases = np.zeros(shape=(num_movies))

    user_embeddings = np.random.normal(loc=0, scale=np.sqrt(1/embedding_dim), size=(num_users, embedding_dim))
    movie_embeddings = np.random.normal(loc=0, scale=np.sqrt(1/embedding_dim), size=(num_movies, embedding_dim))


    train_losses, test_losses, train_errors, test_errors, model =  train(data_by_user_user_index_offsets_train, 
                                                                    data_by_user_movie_indices_train,
                                                                    data_by_user_ratings_train,
                                                                    data_by_user_user_index_offsets_test,
                                                                    data_by_user_movie_indices_test,
                                                                    data_by_user_ratings_test,
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
    
    plot_errors_and_losses(train_losses, test_losses, train_errors, test_errors)
