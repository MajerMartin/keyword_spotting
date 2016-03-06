__author__ = 'martin.majer'

import os
import sys
import argparse
import numpy as np

import dtw as dtw
import common as common


def get_weights(step):
    """
    Get all possible weights combinations for brute-force search.
    :param step: step size (float)
    :return: list of lists with parameters
    """
    weights_temp = [0, 0, 1]
    weights = [[0, 0, 1]]

    i = 1

    while weights_temp != [1, 0, 0]:
        # decrease gamma
        weights_temp[2] = 1 - i * step

        for j in xrange(0, i + 1):
            # increase alpha, decrease beta
            weights_temp[0] = j * step
            weights_temp[1] = abs(1 - weights_temp[2]) - j * step

            # clean weights and append
            weights.append([abs(round(weight, 3)) for weight in weights_temp])

        i += 1

    return weights


def dump_info(weights, global_accuracy, sorted_accuracies, dump_file):
    """
    Save best parameters setting to .txt file.
    :param weights: list of lists with parameters setting
    :param global_accuracy: fold accuracies
    :param sorted_accuracies: sorted fold accuracies
    :param dump_file: output file name
    :return: nothing
    """
    with open(dump_file, 'w') as fw:
        # redirect output
        sys.stdout = fw
        for i in xrange(len(sorted_accuracies)):
            print '{0}{1}:{2: >10}% at {3}'.format('Rank #', i + 1, round(global_accuracy[sorted_accuracies[i]], 2),
                                                   weights[sorted_accuracies[i]])


parser = argparse.ArgumentParser()
parser.add_argument('samples_dir', help='directory containing segmented samples')
parser.add_argument('features_dir', help='directory containing .hdf5 files')
parser.add_argument('input_file', help='.hdf5 input files containing features')
parser.add_argument('output_dir', help='output directory')
parser.add_argument('against', choices=['single', 'all'], help='reference against same speaker or all others')

args = parser.parse_args()

# create path to hdf5 files
path = os.path.join(args.features_dir, args.input_file)

# check whether input is valid and get features type
assert os.path.isfile(path), 'file does not exist'
features_type, extension = os.path.splitext(args.input_file)
assert extension == '.hdf5', 'invalid input file type'

# create output file name
if args.against == 'single':
    ofile_name = os.path.join(args.output_dir, 'params_single.txt')
else:
    ofile_name = os.path.join(args.output_dir, 'params_all.txt')

# load all data
data, speakers, references, ref_keys, test_keys_single, test_keys_all = common.load_all(args.features_dir,
                                                                                        args.samples_dir)

# get distance function parameters
weights = get_weights(0.1)

# get accuracies for all parameters
global_accuracy = []

for params in weights:
    print '{0: <30}{1}'.format('Parameters setting:', params)

    # set up distance function
    alpha = params[0]
    beta = params[1]
    gamma = params[2]

    dist_func = lambda x, y, i, j: alpha * (np.abs(x[i][0] - y[j][0])) + beta * (
        np.abs(x[i][1] - y[j][1])) + gamma * (np.abs(x[i][2] - y[j][2]))

    # average across all speakers
    avg_speakers_accuracy = 0

    for speaker in speakers.keys():
        # get reference and test keys for speaker
        speaker_ref_keys = ref_keys[speaker]

        if args.against == 'single':
            speaker_test_keys = test_keys_single[speaker]
        else:
            speaker_test_keys = test_keys_all[speaker]

        # get features for reference and test samples
        ref_features = common.get_features(data, features_type, speaker_ref_keys)
        test_features = common.get_features(data, features_type, speaker_test_keys)

        # calculate distance matrix
        rows = len(ref_features)
        columns = len(test_features)

        cost_matrix = np.zeros((rows, columns), dtype=np.float64)

        for i in xrange(rows):
            for j in xrange(columns):
                cost_matrix[i, j] = dtw.get_cost(ref_features[i], test_features[j], dist_func)[0]

        speaker_accuracy = common.calculate_accuracy(cost_matrix, speaker_ref_keys, speaker_test_keys, references)
        print '\t{0: >9}{1}{2: <15}{3}{4}'.format('Speaker ', speaker, ':', round(speaker_accuracy, 2), '%')

        avg_speakers_accuracy += speaker_accuracy

    # append average across all speakers
    avg_speakers_accuracy /= len(speakers.keys())
    print '{0: <30}{1}{2}\n'.format('Accuracy:', round(avg_speakers_accuracy, 2), '%')

    global_accuracy.append(avg_speakers_accuracy)

# reversed argsort
sorted_accuracies = np.argsort(global_accuracy)[::-1]

print ''
for i in xrange(5):
    print '{0}{1}:{2: >10}% at {3}'.format('Rank #', i + 1, round(global_accuracy[sorted_accuracies[i]], 2),
                                           weights[sorted_accuracies[i]])
print ''

dump_info(weights, global_accuracy, sorted_accuracies, ofile_name)
