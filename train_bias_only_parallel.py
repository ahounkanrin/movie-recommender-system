import numpy as np
import polars as pl
import os

from data_parser import parse_data, random_split
from tqdm import tqdm
from matplotlib import pyplot as plt


# DATA_DIR = "./data/ml-latest-small"
# DATA_DIR = "./data/ml-25m"
DATA_DIR = "./data/ml-32m"
data = pl.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
# data = data.sort("timestamp")

data_by_user, data_by_movie, index_to_user_id, index_to_movie_id = parse_data(data)
data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test = random_split(data_by_user, data_by_movie)

num_users = len(data_by_user)
num_movies = len(data_by_movie)

user_biases = np.zeros(shape=(num_users))
movie_biases = np.zeros(shape=(num_movies))
lambda_ = 0.1
gamma_ = 0.01
num_epochs = 10

train_losses = []
test_losses = []
train_errors = []
test_errors = []

for epoch in range(num_epochs):
    # update user biases
    for m in range(num_users):
        bias = 0
        movie_counter = 0
        for (n, r) in data_by_user_train[m]:
            bias += lambda_ * (r - movie_biases[n])
            movie_counter += 1
        bias /= (lambda_ * movie_counter + gamma_)
        user_biases[m] = bias

    # update movie biases
    for n in range(num_movies):
        bias = 0
        user_counter = 0
        for (m, r)  in data_by_movie_train[n]:
            bias += lambda_ * (r - user_biases[m])
            user_counter += 1
        bias /= (lambda_ * user_counter + gamma_)
        movie_biases[n] = bias

    # compute train loss
    loss_train = 0
    rmse_train = 0
    rating_counter_train = 0
    for m in range(num_users):
        for (n, r) in data_by_user_train[m]:
            loss_train += (lambda_/2) * (r - user_biases[m] - movie_biases[n])**2
            rmse_train += (r - user_biases[m] - movie_biases[n])**2
            rating_counter_train += 1
    
    loss_train += (gamma_/2) * (np.sum(user_biases**2) + np.sum(movie_biases**2)) 
    # loss_train /= rating_counter_train
    rmse_train = np.sqrt(rmse_train/rating_counter_train)

    # compute test loss
    loss_test = 0
    rating_counter_test = 0
    rmse_test = 0
    for m in range(num_users):
        for (n, r) in data_by_user_test[m]:
            loss_test += (lambda_/2) * (r - user_biases[m] - movie_biases[n])**2
            rmse_test += (r - user_biases[m] - movie_biases[n])**2
            rating_counter_test += 1

    loss_test += (gamma_/2) * (np.sum(user_biases**2) + np.sum(movie_biases**2))
    # loss_test /= rating_counter_test
    rmse_test = np.sqrt(rmse_test/rating_counter_test)
    
    train_losses.append(loss_train)
    test_losses.append(loss_test)
    train_errors.append(rmse_train)
    test_errors.append(rmse_test)

    print(f"Epoch: {epoch+1} \t train_loss = {loss_train:.4f} \t test_loss = {loss_test:.4f}\
          \t mse_train = {rmse_train:.4f} \t mse_test = {rmse_test:.4f}")
    
fig, ax = plt.subplots(2, 1)
ax[0].plot(train_losses, label="Train", color="b")
ax[1].plot(test_losses, label="Test", color="r")
ax[0].legend()
ax[1].legend()
fig.suptitle("Negative log likelihood")
plt.savefig("./outputs/plots/bias_only_model_nll_32M.pdf")
plt.close()

fig, ax = plt.subplots(2, 1)
ax[0].plot(train_errors, label="Train", color="b")
ax[1].plot(test_errors, label="Test", color="r")
ax[0].legend()
ax[1].legend()
plt.suptitle("RMSE")
plt.savefig("./outputs/plots/bias_only_model_rmse_32M.pdf")