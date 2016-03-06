__author__ = 'martin.majer'

import os
import argparse
import numpy as np

import dtw as dtw
import common as common
import svm_utilities as svm


def calculate_train_matrix(ref_features, dist_func):
    """
    Calculate train matrix with distances (Gram matrix).
    :param features: list with reference features
    :param dist_func: distance function
    :return: numpy array with train matrix
    """
    dim = len(ref_features)

    print '\t{0: <25}{1}x{2}'.format('Dimension - train:', dim, dim)

    train_matrix = np.zeros((dim, dim), dtype=np.float64)

    # upper triangle
    for i in xrange(0, dim):
        for j in xrange(i, dim):
            train_matrix[i, j] = dtw.get_cost(ref_features[i], ref_features[j], dist_func=dist_func)[0]

    # get lower triangle
    train_matrix = np.triu(train_matrix).T + np.triu(train_matrix)

    return train_matrix


def calculate_test_matrix(ref_features, test_features, dist_func):
    """
    Calculate test matrix with distances.
    :param ref_features: list with reference features
    :param test_features: list with test features
    :param dist_func: distance function
    :return: numpy array with test matrix
    """
    rows = len(test_features)
    columns = len(ref_features)

    print '\t{0: <25}{1}x{2}'.format('Dimension - test:', rows, columns)

    test_matrix = np.zeros((rows, columns), dtype=np.float64)

    for i in xrange(rows):
        for j in xrange(columns):
            test_matrix[i, j] = dtw.get_cost(ref_features[j], test_features[i], dist_func)[0]

    return test_matrix


parser = argparse.ArgumentParser()
parser.add_argument('samples_dir', help='directory containing segmented samples')
parser.add_argument('features_dir', help='directory containing .hdf5 files')
parser.add_argument('input_files', nargs='+', help='.hdf5 input files containing features')
parser.add_argument('output_dir', help='output directory')
parser.add_argument('against', choices=['single', 'all'], help='reference against same speaker or all others')
parser.add_argument('--speakers', nargs='+', help='speakers numbers', default=['1', '2', '3', '4', '5', '6'], type=str)
parser.add_argument('--inverse', choices=['True', 'False'], help='flip ref/test keys for train/test matrices',
                    default=False)
parser.add_argument('--dist_func', choices=['True', 'False'],
                    help='custom distance function for ste_sti_stzcr (file name must contain "ste_sti_stzcr")',
                    default=False)
parser.add_argument('--alpha', help='alpha parameter for distance function', default=1.0, type=float)
parser.add_argument('--beta', help='beta parameter for distance function', default=1.0, type=float)
parser.add_argument('--gamma', help='gamma parameter for distance function', default=1.0, type=float)

args = parser.parse_args()

# create paths to hdf5 files
paths = []

for file in args.input_files:
    path = os.path.join(args.features_dir, file)
    paths.append(path)

# check whether input is valid
for file in paths:
    assert os.path.isfile(file), 'file does not exist'
    ifile_name, extension = os.path.splitext(file)
    assert extension == '.hdf5', 'invalid input file type'

# load all data
data, speakers, references, ref_keys_tmp, test_keys_single = common.load_all(args.features_dir, args.samples_dir)[0:5]

for file in args.input_files:
    # extract feature type from file name
    features_type, extension = os.path.splitext(file)

    # prepare output file name
    ofile_name = os.path.join(args.output_dir, file)

    # use custom distance function
    if args.dist_func and 'ste_sti_stzcr' in file:
        dist_func = lambda x, y, i, j: args.alpha * (np.abs(x[i][0] - y[j][0])) + args.beta * (
            np.abs(x[i][1] - y[j][1])) + args.gamma * (np.abs(x[i][2] - y[j][2]))
    else:
        dist_func = lambda x, y, i, j: np.linalg.norm(x[i] - y[j])

    print '{0: <25}{1}'.format('Processing file:', file)

    # calculate distance matrix
    if args.against == 'all':
        print '\tAll speakers'

        # get all reference keys for all speakers
        ref_keys = []

        for key in ref_keys_tmp.keys():
            ref_keys.extend(ref_keys_tmp[key])

            # get all test keys for all speakers
            test_keys = []

        for key in test_keys_single.keys():
            test_keys.extend(test_keys_single[key])

        if args.inverse:
            ref_keys, test_keys = test_keys, ref_keys

        # get features for reference and test samples
        ref_features = common.get_features(data, features_type, ref_keys)
        test_features = common.get_features(data, features_type, test_keys)

        # calculate Gram matrix for SVM training
        train_matrix = calculate_train_matrix(ref_features, dist_func)

        # calculate distance matrix for SVM prediction
        test_matrix = calculate_test_matrix(ref_features, test_features, dist_func)

        # save to hdf5 file
        svm.save_distance_matrix(ofile_name, 'train', args.against, train_matrix, ref_keys, ref_keys)
        svm.save_distance_matrix(ofile_name, 'test', args.against, test_matrix, ref_keys, test_keys)

        print ''
    else:
        for speaker in args.speakers:
            print '\t{0: <25}{1}, {2}'.format('Speaker:', speaker, args.against)

            # get reference and test keys for speaker
            ref_keys = ref_keys_tmp[speaker]
            test_keys = test_keys_single[speaker]

            if args.inverse:
                ref_keys, test_keys = test_keys, ref_keys

            # get features for reference and test samples
            ref_features = common.get_features(data, features_type, ref_keys)
            test_features = common.get_features(data, features_type, test_keys)

            # calculate Gram matrix for SVM training
            train_matrix = calculate_train_matrix(ref_features, dist_func)

            # calculate distance matrix for SVM prediction
            test_matrix = calculate_test_matrix(ref_features, test_features, dist_func)

            # save to hdf5 file
            svm.save_distance_matrix(ofile_name, speaker + '/train', args.against, train_matrix, ref_keys, ref_keys)
            svm.save_distance_matrix(ofile_name, speaker + '/test', args.against, test_matrix, ref_keys, test_keys)

            print ''
