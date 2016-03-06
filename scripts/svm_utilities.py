__author__ = 'martin.majer'

import h5py
import numpy as np
from sklearn import svm

import common as common


def get_targets(references, keys, speaker):
    """
    Get numeric values for keys.
    :param references: dictionary with references
    :param keys: list with keys
    :param speaker: speaker key
    :return: list
    """
    targets = []

    for key in keys[speaker]:
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


def get_accuracy_data(dm_dir, samples_dir, type, speakers, speakers_train, speakers_test):
    """
    Get accuracies for all speakers.
    :param dm_dir: directory with .hdf5 files (distance matrices)
    :param samples_dir: directory with .wav files (segmented samples)
    :param type: against single or all speakers ['single', 'all']
    :param speakers: list with speaker numbers (e.g. [1, 2, ...])
    :param speakers_train: list with keys to train data
    :param speakers_test: list with keys to test data
    :return: dictionary with accuracies per method
    """
    accuracy_data = {}

    # get paths to all files with distance matrices
    file_paths = common.get_files(dm_dir, '.hdf5')[1]

    # get references for all samples
    references = common.load_references(samples_dir)

    # for all feature types
    for file_name, file_path in file_paths.iteritems():
        # load all train and test data
        train_matrices, train_ref_keys, train_test_keys = load_distance_matrices(file_path, type, speakers_train)
        test_matrices, test_ref_keys, test_test_keys = load_distance_matrices(file_path, type, speakers_test)

        feature_accuracies = []

        # fit and predict for all speakers
        for speaker in speakers:
            speaker_train = speakers_train[speaker - 1]
            speaker_test = speakers_test[speaker - 1]

            # get targets and references values
            y = get_targets(references, train_ref_keys, speaker_train)
            ref = get_targets(references, test_test_keys, speaker_test)

            # get train and test matrix
            X = train_matrices[speaker_train]
            z = test_matrices[speaker_test]

            pred = fit_and_predict(X, y, z)
            acc = calculate_accuracy(pred, ref)

            feature_accuracies.append(acc)

        # calculate mean across all speakers
        if type == 'single':
            feature_accuracies.append(np.mean(feature_accuracies))

        # remove .hdf5 from file name and save
        accuracy_data[file_name[:-5]] = feature_accuracies

    return accuracy_data


def save_distance_matrix(ofile_name, speaker, type, distance_matrix, ref_keys, test_keys):
    """
    Save distance matrix to .hdf5 file.
    :param ofile_name: output file name
    :param speaker: speaker number (used as key with type)
    :param type: against single or all speakers (used as key with speaker number)
    :param distance_matrix: numpy array with distances
    :param ref_keys: list of reference keys
    :param test_keys: list of test keys
    :return: nothing
    """
    speaker_key = '/' + type + '/' + speaker

    with h5py.File(ofile_name, 'a') as fw:
        if speaker_key in fw:
            print '\n\tOverwriting...'
            del fw[speaker_key]

        group = fw.create_group(speaker_key)
        group['data'] = distance_matrix
        group['ref'] = ref_keys
        group['test'] = test_keys


def load_distance_matrices(path, type, speakers):
    """
    Load distance matrices, reference and test keys from .hdf5 file.
    :param path: path to .hdf5 file
    :param type: against single or all speakers ['single', 'all']
    :param speakers: list with speaker numbers
    return: dictionaries with distance matrices, reference and test keys
    """
    distance_matrices = {}
    ref_keys = {}
    test_keys = {}

    with h5py.File(path, 'r') as fr:
        group = fr[type]

        for speaker in speakers:
            distance_matrices[speaker] = group[str(speaker)]['data'].value
            ref_keys[speaker] = group[str(speaker)]['ref'].value
            test_keys[speaker] = group[str(speaker)]['test'].value

    return distance_matrices, ref_keys, test_keys
