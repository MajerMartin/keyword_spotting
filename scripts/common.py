__author__ = 'martin.majer'

import os
import re
import h5py
import pickle
import numpy as np
import pandas as pd


def get_files(data_dir, extension, verbose=False):
    """
    Extract all file names and their paths with specified extension from directory.
    :param data_dir: directory with data
    :param extension: extension, e.g. '.txt'
    :param verbose: print to console
    :return: list with file names and dictionary with paths
    """
    file_names = []
    file_paths = {}

    assert os.path.isdir(data_dir), 'input directory does not exist'

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(extension):
                file_names.append(file)
                file_paths[file] = os.path.join(data_dir, file)

    if verbose:
        print '{0: <25}{1}'.format('Files count:', len(file_paths))
        print '{0: <25}{1}'.format('Names sample:', file_names[:3])
        print '{0: <25}{1}'.format('Path sample:', file_paths['1-1-1.wav'])
        print ''

    return file_names, file_paths


def load_features_data(features_dir):
    """
    Load all features from .hdf5 files.
    :param features_dir: directory containing .hdf5 files
    :return: dictionary with data for all feature type
    """
    # get all feature containers and their paths
    file_names, file_paths = get_files(features_dir, '.hdf5')

    data = {}

    for file_name in file_names:
        # remove .hdf5 extension for dictionary key
        feature_key = file_name[:-5]

        data[feature_key] = {}

        with h5py.File(file_paths[file_name], 'r') as fr:
            file_keys = fr.keys()
            for file_key in file_keys:
                data[feature_key][file_key] = fr[file_key].value

    return data


def load_speakers_data(samples_dir):
    """
    Assign samples to their speakers. First letter in file name must identify speaker.
    :param samples_dir: directory containing .wav files
    :return: dictionary with assigned samples
    """
    file_names = get_files(samples_dir, '.wav')[0]

    speakers = {}

    for file_name in file_names:
        # append to list in dictionary
        speakers.setdefault(file_name[0], []).append(file_name)

    return speakers


def load_references(samples_dir):
    """
    Assign references to samples. Last character in file name must be number which has been spoken.
    :param samples_dir: directory containing .wav files
    :return: dictionary with assigned references
    """
    file_names = get_files(samples_dir, '.wav')[0]

    references = {}

    for file_name in file_names:
        references[file_name] = file_name[-5]

    return references


def ref_test_split(samples_dir):
    """
    Split samples into reference a test for every speaker.
    :param samples_dir: directory containing .wav files
    :return: dictionaries with reference and test keys (single/all)
    """
    speakers = load_speakers_data(samples_dir)
    file_names = get_files(samples_dir, '.wav')[0]

    ref_keys = {}
    test_keys_single = {}
    test_keys_all = {}

    # get reference keys for single speaker
    for key, value in speakers.iteritems():
        ref_keys[key] = value[:10]
        test_keys_single[key] = value[10:]

    # get reference keys for all speakers
    for file in file_names:
        for speaker in speakers.keys():
            if not re.match(r'%s-1-\d\.wav' % (speaker), file):
                test_keys_all.setdefault(speaker, []).append(file)

    return ref_keys, test_keys_single, test_keys_all


def load_all(features_dir, samples_dir):
    """
    Load features, speakers, references, reference and test keys.
    :param features_dir: directory containing .hdf5 files
    :param samples_dir: directory containing .wav files
    :return: directories with data, speakers, references, reference and test keys
    """
    data = load_features_data(features_dir)
    speakers = load_speakers_data(samples_dir)
    references = load_references(samples_dir)
    ref_keys, test_keys_single, test_keys_all = ref_test_split(samples_dir)

    return data, speakers, references, ref_keys, test_keys_single, test_keys_all


def get_features(data, features_type, keys):
    """
    Get features for selected files.
    :param data: dictionary with all features
    :param features_type: features type
    :param keys: list with keys
    :return: list with features
    """
    features = []

    for key in keys:
        features.append(data[features_type][key])

    return features


def save_distance_matrix(ofile_name, distance_matrix, ref_keys, test_keys):
    """
    Save distance matrix to .dat file.
    :param ofile_name: output file name
    :param distance_matrix: dictionary with distances
    :param ref_keys: list of reference keys
    :param test_keys: list of test keys
    :return: nothing
    """
    with open(ofile_name + '.dat', 'wb') as fw:
        pickle.dump([distance_matrix, ref_keys, test_keys], fw, protocol=pickle.HIGHEST_PROTOCOL)


def load_distance_matrix(filename):
    """
    Load distance matrices, reference and test keys from .dat file.
    :param filename: .dat file name
    :return: dictionary with distance matrices, list with reference and test keys
    """
    with open(filename, 'rb') as fr:
        data = pickle.load(fr)

    return data[0], data[1], data[2]


def get_data(dm_dir, feature_type):
    """
    Get data for one feature type.
    :param dm_dir: directory with .dat files
    :param feature_type: type of feature
    :return: dictionary with distance matrices, list with reference and test keys
    """
    file_paths = get_files(dm_dir, '.dat')[1]
    distance_matrix = ref_keys = test_keys = None

    for file_name, file_path in file_paths.iteritems():
        if feature_type == os.path.basename(file_name).split('.')[0]:
            distance_matrix, ref_keys, test_keys = load_distance_matrix(file_path)

    return distance_matrix, ref_keys, test_keys


def get_speaker_keys(keys, speaker):
    """
    Get keys for speaker.
    :param keys: list with all keys
    :param speaker: speakers number
    :return: list with speakers keys
    """
    selection = []

    for key in keys[speaker]:
        if key.startswith(speaker + '-'):
            selection.append(key)

    return selection


def compute_accuracy(ref_keys, recognition, references):
    """
    Calculate recognition accuracy.
    :param ref_keys: list with reference keys
    :param recognition: list with recognized keys
    :param references: dictionary with references
    :return: accuracy
    """
    correct = 0
    wrong = 0

    for ref, rec in zip(ref_keys, recognition):
        if references[ref] == references[rec]:
            correct += 1
        else:
            wrong += 1

    accuracy = 100.0 * correct / float(correct + wrong)

    return accuracy


def recognize(test_keys, ref_keys, distance_matrix):
    """

    :param test_keys: list with test keys
    :param ref_keys: list with reference keys
    :param distance_matrix: dictionary with distances
    :return: list with recognized keys
    """
    recognition = []

    for test_key in test_keys:
        min_value = 99999999999
        best = None

        for ref_key in ref_keys:
            distance = distance_matrix[ref_key, test_key]
            if distance < min_value:
                min_value = distance
                best = ref_key

        recognition.append(best)

    return recognition


def calculate_accuracy(distance_matrix, ref_keys, test_keys, references):
    """
    Calculate recognition accuracy (used for optimization).
    :param distance_matrix: numpy array with distances
    :param ref_keys: list with reference keys
    :param test_keys: list with test keys
    :param references: dictionary with references
    :return: accuracy
    """
    correct = 0
    wrong = 0

    test_keys_count = len(test_keys)

    for col in xrange(test_keys_count):
        # get distances for test sample to reference samples
        ref_distance = distance_matrix[:, col]

        # find index of minimum distance to reference sample
        nearest = np.argmin(ref_distance)

        if references[test_keys[col]] == references[ref_keys[nearest]]:
            correct += 1
        else:
            wrong += 1

    accuracy = 100.0 * correct / float(correct + wrong)

    return accuracy


def create_dataframe(accuracy_data, keys):
    """
    Create pandas dataframe.
    :param accuracy_data: dictionary with accuracies per method
    :param keys: list with column names
    :return: pandas dataframe
    """
    df = pd.DataFrame.from_dict(accuracy_data, orient='index')
    df.columns = keys
    df.sort(axis=0, inplace=True)

    return df
