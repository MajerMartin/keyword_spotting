__author__ = 'martin.majer'

import os
import sys
import h5py
import argparse
import numpy as np
import features as fts
from scipy.io.wavfile import read

import common as common

DEBUG = False


def get_args(argvals=None):
    """
    Parse arguments from command-line.
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='features')

    # create subparsers for features
    parser_ste = subparsers.add_parser('ste', help='short-term energy')
    parser_sti = subparsers.add_parser('sti', help='short-term intensity')
    parser_stzcr = subparsers.add_parser('stzcr', help='short-term zero-crossing rate')
    parser_ste_sti_stzcr = subparsers.add_parser('ste_sti_stzcr',
                                                 help='matrix containing ste, sti, stzcr in columns (help)')
    parser_fb_en = subparsers.add_parser('fb_en', help='filterbank energies (help)')
    parser_log_fb_en = subparsers.add_parser('log_fb_en', help='log-filterbank energies (help)')
    parser_mfcc = subparsers.add_parser('mfcc', help='mel-frequency cepstral coefficients (help)')

    # add input directory and output file argument
    parser.add_argument('input_dir', help='directory containing .wav files')
    parser.add_argument('output_file', help='output file (.hdf5)')

    # add arguments for framing
    parser.add_argument('frame_length', help='frame length in seconds', type=float)
    parser.add_argument('frame_step', help='frame step size in seconds', type=float)
    parser.add_argument('--win_func', choices=['hamming', 'hanning'],
                        help='type of window function, default=rectangular',
                        default='rectangular')

    # add argument for normalization
    parser.add_argument('--norm', choices=['True', 'False'], help='features normalization, default=False',
                        default=False)

    # add feature specific arguments
    parser_ste_sti_stzcr.add_argument('--deltas', choices=['True', 'False'], help='append deltas, default=False',
                                      default=False)
    parser_ste_sti_stzcr.add_argument('--ddeltas', choices=['True', 'False'], help='append delta-deltas, default=False',
                                      default=False)

    parser_fb_en.add_argument('--filter_count', help='number of filters, default=26', type=int, default=26)
    parser_fb_en.add_argument('--fft_size', help='n-points of discrete fourier transformation, default=512', type=int,
                              default=512)
    parser_fb_en.add_argument('--low_freq', help='lowest filter frequency bound, default=0', type=int, default=0)
    parser_fb_en.add_argument('--high_freq', help='highest filter frequency bound, default=samplerate/2', type=int,
                              default=None)
    parser_fb_en.add_argument('--deltas', choices=['True', 'False'], help='append deltas, default=False', default=False)
    parser_fb_en.add_argument('--ddeltas', choices=['True', 'False'], help='append delta-deltas, default=False',
                              default=False)

    parser_log_fb_en.add_argument('--filter_count', help='number of filters, default=26', type=int, default=26)
    parser_log_fb_en.add_argument('--fft_size', help='n-points of discrete fourier transformation, default=512',
                                  type=int,
                                  default=512)
    parser_log_fb_en.add_argument('--low_freq', help='lowest filter frequency bound, default=0', type=int, default=0)
    parser_log_fb_en.add_argument('--high_freq', help='highest filter frequency bound, default=samplerate/2', type=int,
                                  default=None)
    parser_log_fb_en.add_argument('--deltas', choices=['True', 'False'], help='append deltas, default=False',
                                  default=False)
    parser_log_fb_en.add_argument('--ddeltas', choices=['True', 'False'], help='append delta-deltas, default=False',
                                  default=False)

    parser_mfcc.add_argument('--coeffs', help='number of coefficients, default=13', type=int, default=13)
    parser_mfcc.add_argument('--filter_count', help='number of filters, default=26', type=int, default=26)
    parser_mfcc.add_argument('--fft_size', help='n-points of discrete fourier transformation, default=512', type=int,
                             default=512)
    parser_mfcc.add_argument('--low_freq', help='lowest filter frequency bound, default=0', type=int, default=0)
    parser_mfcc.add_argument('--high_freq', help='highest filter frequency bound, default=samplerate/2', type=int,
                             default=None)
    parser_mfcc.add_argument('--deltas', choices=['True', 'False'], help='append deltas, default=False', default=False)
    parser_mfcc.add_argument('--ddeltas', choices=['True', 'False'], help='append delta-deltas, default=False',
                             default=False)

    return parser.parse_args(argvals)


def report(args):
    """
    Print table with arguments.
    :param args: parsed arguments
    :return: nothing
    """
    print '{0: <25}{1}'.format('Input directory:', args.input_dir)
    print '{0: <25}{1}'.format('Output file:', args.output_file)
    print '{0: <25}{1}'.format('Frame length:', args.frame_length)
    print '{0: <25}{1}'.format('Frame step size:', args.frame_step)
    print '{0: <25}{1}'.format('Window function:', args.win_func)
    print '{0: <25}{1}'.format('Features:', args.features)
    print '{0: <25}{1}'.format('Normalization:', args.norm)

    if args.features == 'ste_sti_stzcr':
        print '{0: <25}{1}'.format('Deltas:', args.deltas)
        print '{0: <25}{1}'.format('Delta-deltas:', args.ddeltas)

    elif args.features == 'fb_en' or args.features == 'log_fb_en':
        print '{0: <25}{1}'.format('Number of filters:', args.filter_count)
        print '{0: <25}{1}'.format('FFT size:', args.fft_size)
        print '{0: <25}{1}'.format('Lowest frequency:', args.low_freq)
        print '{0: <25}{1}'.format('Highest frequency:', args.high_freq)
        print '{0: <25}{1}'.format('Deltas:', args.deltas)
        print '{0: <25}{1}'.format('Delta-deltas:', args.ddeltas)


    elif args.features == 'mfcc':
        print '{0: <25}{1}'.format('Coefficients:', args.coeffs)
        print '{0: <25}{1}'.format('Number of filters:', args.filter_count)
        print '{0: <25}{1}'.format('FFT size:', args.fft_size)
        print '{0: <25}{1}'.format('Lowest frequency:', args.low_freq)
        print '{0: <25}{1}'.format('Highest frequency:', args.high_freq)
        print '{0: <25}{1}'.format('Deltas:', args.deltas)
        print '{0: <25}{1}'.format('Delta-deltas:', args.ddeltas)

    print ''


def normalize(features):
    """
    Normalize features.
    :param features: numpy array with features (vector or matrix)
    :return: numpy array with normalized features
    """
    if isinstance(features, list):
        features = np.array(features, dtype=np.float64)
        return fts.stand(features)
    else:
        # normalize columns
        features = features.T
        norm_matrix = np.zeros(shape=features.shape)

        for i, row in enumerate(features):
            norm_matrix[i] = fts.stand(row)

        return norm_matrix.T


def save(file_names, features, output_file):
    """
    Save features to HDF5 container.
    :param file_names: list with file names of samples
    :param features: list with features of samples
    :param output_file: output file name
    :return: nothing
    """
    ofile_name, ofile_extension = os.path.splitext(output_file)

    assert len(file_names) == len(features)
    assert ofile_extension == '.hdf5', 'invalid output file type'

    with h5py.File(output_file, 'w') as fw:
        for i, sample in enumerate(file_names):
            fw[sample] = features[i]


def dump_info(args):
    """
    Save table with arguments to .txt file.
    :param args: parsed arguments
    :return: nothing
    """
    ofile_name, ofile_extension = os.path.splitext(args.output_file)
    dump_file = ofile_name + '_info.txt'

    with open(dump_file, 'w') as fw:
        # redirect output
        sys.stdout = fw
        report(args)


# parameters for debugging
if DEBUG:
    argvals = 'mfcc data/segmented feats.hdf5 0.01 0.005 --win_func hamming'.split()
else:
    argvals = None

# parse arguments and print them
args = get_args(argvals)
report(args)

# collect files
file_names, file_paths = common.get_files(args.input_dir, '.wav', verbose=True)

# select window function for framing
if args.win_func == 'rectangular':
    win_func = lambda x: np.ones((x,))
elif args.win_func == 'hamming':
    win_func = lambda x: np.hamming(x)
elif args.win_func == 'hanning':
    win_func = lambda x: np.hanning(x)

# select feature function
if args.features == 'ste':
    feat_func = lambda x, y: fts.get_ste(x, y, args.frame_length, args.frame_step, win_func)
elif args.features == 'sti':
    feat_func = lambda x, y: fts.get_sti(x, y, args.frame_length, args.frame_step, win_func)
elif args.features == 'stzcr':
    feat_func = lambda x, y: fts.get_stzcr(x, y, args.frame_length, args.frame_step, win_func)
elif args.features == 'ste_sti_stzcr':
    feat_func = lambda x, y: fts.get_ste_sti_stzcr(x, y, args.frame_length, args.frame_step, win_func)
elif args.features == 'fb_en':
    feat_func = lambda x, y: fts.get_filterbank_energies(x, y, args.frame_length, args.frame_step, win_func,
                                                         args.filter_count, args.fft_size, args.low_freq,
                                                         args.high_freq, False)
elif args.features == 'log_fb_en':
    feat_func = lambda x, y: fts.get_filterbank_energies(x, y, args.frame_length, args.frame_step, win_func,
                                                         args.filter_count, args.fft_size, args.low_freq,
                                                         args.high_freq, True)
elif args.features == 'mfcc':
    feat_func = lambda x, y: fts.get_mfcc(x, y, args.frame_length, args.frame_step, win_func, args.coeffs,
                                          args.filter_count, args.fft_size, args.low_freq, args.high_freq)

# calculate selected features
features = []

if args.features in ['ste_sti_stzcr', 'mfcc', 'fb_en', 'log_fb_en']:
    deltas = None
    ddeltas = None

    for i, sample in enumerate(file_names):
        if i % 10 == 0:
            print '\r{0: <25}{1}'.format('Processing sample:', sample),

        rate, signal = read(file_paths[sample])

        sample_features = feat_func(signal, rate)

        # calculate deltas or delta-deltas
        if args.deltas:
            deltas = fts.get_deltas(sample_features, axis=0, order=1)
        if args.ddeltas:
            ddeltas = fts.get_deltas(sample_features, axis=0, order=2)

        # append deltas or delta-deltas to features
        if deltas is not None:
            sample_features = np.hstack((sample_features, deltas))
        if ddeltas is not None:
            sample_features = np.hstack((sample_features, ddeltas))

        # normalize features
        if args.norm:
            sample_features = normalize(sample_features)

        features.append(sample_features)
else:
    for i, sample in enumerate(file_names):
        if i % 10 == 0:
            print '\r{0: <25}{1}'.format('Processing sample:', sample),

        rate, signal = read(file_paths[sample])

        sample_features = feat_func(signal, rate)

        # normalize features
        if args.norm:
            sample_features = normalize(sample_features)

        features.append(sample_features)

print '\r{0: <25}{1}'.format('Processing sample:', sample),
print '\n{0: <25}{1}'.format('Samples processed:', len(features))

# save features to file and dump features info
save(file_names, features, args.output_file)
dump_info(args)
