import numpy as np
from matplotlib import pyplot as plt


def plot_dist(lambda_, tau, r):
    u = np.arange(-5, 5, 0.25)
    v = np.arange(-5, 5, 0.25)
    U, V = np.meshgrid(u, v)

    P = np.exp(- 0.5 * tau * (U**2 + V**2) - 0.5 * lambda_ * (r - U * V)**2)
    plt.axis("equal")
    plt.xlabel("u")
    plt.ylabel("v")

    surf = plt.contourf(U, V, P)
    plt.show()


plot_dist(1, 0.01, 2.5)