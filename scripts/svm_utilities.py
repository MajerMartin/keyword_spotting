__author__ = 'martin.majer'

import numpy as np
from sklearn import svm


def dict_to_array(ref_keys, test_keys, distance_matrix):
    """
    Convert dictionary to numpy array.
    :param ref_keys: list with reference keys
    :param test_keys: list with test keys
    :param distance_matrix: dictionary representing distance matrix
    :return: numpy array
    """
    rows = len(test_keys)
    cols = len(ref_keys)
    matrix = np.zeros((rows, cols), dtype=np.float64)

    for i, test in enumerate(test_keys):
        for j, ref in enumerate(ref_keys):
            matrix[i][j] = 1.0 - distance_matrix[ref, test]

    return matrix


def get_targets(keys, references):
    """
    Get numeric values for keys.
    :param keys: list with keys
    :param references: dictionary with references
    :return: list
    """
    targets = []

    for key in keys:
        targets.append(references[key])

    return targets


def calculate_accuracy(prediction, references):
    """
    Calculate recognition accuracy.
    :param prediction: numpy array with predicted values
    :param references: numpy array with actual values
    :return: accuracy in float
    """
    # must be int (svm returns strings)
    prediction = np.asarray(prediction).astype(int)
    references = np.asarray(references).astype(int)

    return 100.0 * np.sum(np.equal(prediction, references)) / len(references)


def fit_and_predict(X, y, z):
    """
    Fit SVM and predict.
    :param X: train matrix
    :param y: correct output
    :param z: test matrix
    :return:
    """
    clf = svm.SVC(kernel='precomputed')
    clf.fit(X, y)

    return clf.predict(z)
