__author__ = 'martin.majer'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_cost(x, y, dist_func=lambda x, y, i, j: np.linalg.norm(x[i] - y[j]), norm=True):
    """
    Calculate optimal warping path cost.
    :param x: reference sequence
    :param y: test sequence
    :param dist_func: distance function, default=euclidean distance;
    4 arguments: x, y, i, j, where x and y are sequences and i and j are indexes
    :param norm: normalization
    :return: cost, global cost matrix
    """
    rows = len(x)
    cols = len(y)

    cost_matrix = np.zeros((rows + 1, cols + 1))

    cost_matrix[:, 0] = np.inf
    cost_matrix[0, :] = np.inf
    cost_matrix[0, 0] = 0

    for i in xrange(rows):
        for j in xrange(cols):
            cost = dist_func(x, y, i, j)
            cost_matrix[i + 1, j + 1] = cost + min(cost_matrix[i, j + 1],
                                                   cost_matrix[i + 1, j],
                                                   cost_matrix[i, j])
    if norm:
        return cost_matrix[rows, cols] / len(x), cost_matrix
    else:
        return cost_matrix[rows, cols], cost_matrix


def get_path(cost_matrix):
    """
    Return optimal warping path.
    :param cost_matrix: numpy array with cost values
    :return: list with optimal warping path points
    """
    i = cost_matrix.shape[0] - 1
    j = cost_matrix.shape[1] - 1

    path = [[j, i]]  # ending point

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if cost_matrix[i - 1, j] == min(cost_matrix[i - 1, j - 1],
                                            cost_matrix[i - 1, j],
                                            cost_matrix[i, j - 1]):
                i -= 1
            elif cost_matrix[i, j - 1] == min(cost_matrix[i - 1, j - 1],
                                              cost_matrix[i - 1, j],
                                              cost_matrix[i, j - 1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append([j, i])

    return path


def plot_matrix(cost_matrix, path=None):
    """
    Plot cost matrix and optimal warping path.
    :param cost_matrix: numpy array with global cost values
    :param path: list with optimal warping path points
    :return: nothing
    """
    plt.imshow(cost_matrix, interpolation='nearest', cmap='Greens')
    plt.gca().invert_yaxis()
    plt.xlabel("Reference sequence")
    plt.ylabel("Test sequence")
    plt.grid()
    plt.colorbar()

    if path is not None:
        x = [point[0] for point in path]
        y = [point[1] for point in path]
        plt.plot(x, y)
