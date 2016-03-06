__author__ = 'martin.majer'

import os
import argparse
import numpy as np

import dtw as dtw
import common as common

parser = argparse.ArgumentParser()
parser.add_argument('samples_dir', help='directory containing segmented samples')
parser.add_argument('features_dir', help='directory containing .hdf5 files')
parser.add_argument('input_files', nargs='+', help='.hdf5 input files containing features')
parser.add_argument('output_dir', help='output directory')
parser.add_argument('--speakers', nargs='+', help='speakers numbers', default=['1', '2', '3', '4', '5', '6'], type=str)
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
data, speakers, references, ref_keys, test_keys_single, test_keys_all = common.load_all(args.features_dir,
                                                                                        args.samples_dir)

for file in args.input_files:
    # extract feature type from file name
    features_type, extension = os.path.splitext(file)

    # prepare output file name
    ofile_name = os.path.join(args.output_dir, features_type)

    # use custom distance function
    if args.dist_func and 'ste_sti_stzcr' in file:
        dist_func = lambda x, y, i, j: args.alpha * (np.abs(x[i][0] - y[j][0])) + args.beta * (
            np.abs(x[i][1] - y[j][1])) + args.gamma * (np.abs(x[i][2] - y[j][2]))
    else:
        dist_func = lambda x, y, i, j: np.linalg.norm(x[i] - y[j])

    print '{0: <25}{1}'.format('Processing file:', file)

    cost_matrix = {}

    # calculate distance matrix for all speakers
    for speaker in args.speakers:
        print '\t{0: <25}{1}'.format('Speaker:', speaker)

        # get reference and test keys for speaker
        speaker_ref_keys = ref_keys[speaker]
        speaker_test_keys = test_keys_all[speaker]

        for ref in speaker_ref_keys:
            for test in speaker_test_keys:
                ref_data = data[features_type][ref]
                test_data = data[features_type][test]
                cost_matrix[ref, test] = dtw.get_cost(ref_data, test_data, dist_func)[0]

    # save to .dat file
    common.save_distance_matrix(ofile_name, cost_matrix, ref_keys, test_keys_all)

    print ''
