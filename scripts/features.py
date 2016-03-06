__author__ = 'martin.majer'

import numpy as np
from scipy.fftpack import dct


def get_frames(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,))):
    """
    Split signal into frames (with our without overlay) and apply window function.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :return: numpy array with single frames
    """
    signal_length = len(signal)

    # convert frame length and step size to samples per frame
    frame_length = int(rate * frame_length)
    frame_step = int(rate * frame_step)

    # calculate frame count
    if signal_length < frame_length:
        frames_count = 1
    else:
        frames_count = 1 + int(np.ceil((signal_length * 1.0 - frame_length) / frame_step))

    # if last frame is incomplete, add padding of zeroes
    padding_length = int((frames_count - 1) * frame_step + frame_length)
    padding = np.zeros((padding_length - signal_length))
    signal = np.concatenate((signal, padding))

    # create array with frame indexes
    indexes = np.tile(np.arange(0, frame_length), (frames_count, 1)) + np.tile(
        np.arange(0, frames_count * frame_step, frame_step), (frame_length, 1)).T

    # use indexes mask to get single frames
    frames = signal[indexes]

    # create window function mask
    windows = np.tile(window_function(frame_length), (frames_count, 1))

    return frames * windows


def get_ste(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,))):
    """
    Compute short-term energy.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :return: list of short-term energy values
    """
    frames = get_frames(signal, rate, frame_length, frame_step, window_function)

    ste = []

    for frame in frames:
        frame_ste = np.sum(frame ** 2)  # signal is numpy array -> np.sum() is faster than sum()
        ste.append(frame_ste)

    return ste


def get_sti(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,))):
    """
    Compute short-term intensity.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :return: list of short-term intensity values
    """
    frames = get_frames(signal, rate, frame_length, frame_step, window_function)

    sti = []

    for frame in frames:
        frame_sti = np.sum(np.abs(frame))
        sti.append(frame_sti)

    return sti


def get_stzcr(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,))):
    """
    Compute short-term zero-crossing rate.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :return: list of short-term zero-crossing rate values
    """
    frames = get_frames(signal, rate, frame_length, frame_step, window_function)

    stzcr = []

    for frame in frames:
        frame_stzcr = 0.5 * np.sum(np.abs(np.diff(np.sign(frame))))
        stzcr.append(frame_stzcr)

    return stzcr


def get_ste_sti_stzcr(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,))):
    """
    Combine short-term energy, short-term intensity and short-term zero-crossing rate.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :return: numpy array with ste, sti, stzcr in columns
    """
    frames = get_frames(signal, rate, frame_length, frame_step, window_function)

    ste = []
    sti = []
    stzcr = []

    for frame in frames:
        frame_ste = np.sum(frame ** 2)
        frame_sti = np.sum(np.abs(frame))
        frame_stzcr = 0.5 * np.sum(np.abs(np.diff(np.sign(frame))))
        ste.append(frame_ste)
        sti.append(frame_sti)
        stzcr.append(frame_stzcr)

    ste_sti_stzcr = np.vstack((ste, sti, stzcr))

    return ste_sti_stzcr.T


def get_wav_time(signal, rate):
    """
    Compute time axis for audio signal.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :return: time
    """
    return np.arange(len(signal)) / float(rate)


def norm(features):
    """
    Features normalization.
    :param features: list of features
    :return: list of normalized features
    """
    return (features - np.min(features)) / (np.max(features) - np.min(features))


def stand(features):
    """
    Features standardization.
    :param features: list of features
    :return: list of standardized features
    """
    return (features - np.mean(features)) / np.std(features)


def hz_to_mel(hz):
    """
    Convert Hz to Mel.
    :param hz: value in Hz
    :return: value in Mel
    """
    return 2595 * np.log10(1 + hz * 1.0 / 700)


def mel_to_hz(mel):
    """
    Convert Mel to Hz.
    :param mel: value in Mel
    :return: value in Hz
    """
    return 700 * (10 ** (mel * 1.0 / 2595) - 1)


def get_power_spectrum(frames, fft_size):
    """
    Calculate periodogram estimate of power spectrum (for each frame).
    :param frames:  framed audio signal
    :param fft_size: n-points of discrete fourier transform
    :return: power spectrum for each frame
    """
    # np.square is element-wise
    return 1.0 / fft_size * np.square(np.abs(np.fft.rfft(frames, fft_size)))


def get_filterbanks(rate, filters_count=26, fft_size=512, low_freq=0, high_freq=None):
    """
    Create Mel-filterbanks.
    :param rate: frame rate of audio signal
    :param filters_count: number of filters
    :param fft_size: n-points of discrete fourier transform
    :param low_freq: start frequency of first filter
    :param high_freq: end frequency of last filter
    :return: numpy array with filterbanks
    """
    if high_freq is None:
        high_freq = rate / 2

    assert high_freq <= rate / 2, "high_freq must be lower than rate / 2"

    # convert Hz to Mel
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)

    # calculate filter points linearly spaced between lowest and highest frequency
    mel_points = np.linspace(low_mel, high_mel, filters_count + 2)

    # convert points back to Hz
    hz_points = mel_to_hz(mel_points)

    # round frequencies to nearest fft bin
    fft_bin = np.floor((fft_size + 1) * hz_points / rate)

    # first filterbank will start at the first point, reach its peak at the second point
    # then return to zero at the 3rd point. The second filterbank will start at the 2nd
    # point, reach its max at the 3rd, then be zero at the 4th etc.
    filterbanks = np.zeros([filters_count, fft_size / 2 + 1])

    for i in xrange(filters_count):
        # from left to peak
        for j in xrange(int(fft_bin[i]), int(fft_bin[i + 1])):
            filterbanks[i, j] = (j - fft_bin[i]) / (fft_bin[i + 1] - fft_bin[i])
        # from peak to right
        for j in xrange(int(fft_bin[i + 1]), int(fft_bin[i + 2])):
            filterbanks[i, j] = (fft_bin[i + 2] - j) / (fft_bin[i + 2] - fft_bin[i + 1])

    return filterbanks


def get_filterbank_energies(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,)),
                            filters_count=26, fft_size=512, low_freq=0, high_freq=None, log=False):
    """
    Compute Mel-filterbank energies.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :param filters_count: number of filters
    :param fft_size: n-points of discrete fourier transform
    :param low_freq: start frequency of first filter
    :param high_freq: end frequency of last filter
    :param log: if True, compute log-filterbank energies
    :return: numpy array with (log-)filterbank energies
    """
    if high_freq is None:
        high_freq = rate / 2

    frames = get_frames(signal, rate, frame_length, frame_step, window_function)

    power_spectrum = get_power_spectrum(frames, fft_size)
    filterbanks = get_filterbanks(rate, filters_count, fft_size, low_freq, high_freq)

    # weighted sum of the fft energies around filterbank frequencies
    filterbank_energies = np.dot(power_spectrum, filterbanks.T)

    if log == True:
        # replace zeroes with machine epsilon to prevent errors in log operation
        filterbank_energies = np.where(filterbank_energies == 0, np.finfo(float).eps, filterbank_energies)
        return np.log(filterbank_energies)
    else:
        return filterbank_energies


def get_mfcc(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,)), coeffs=13,
             filters_count=26, fft_size=512, low_freq=0, high_freq=None):
    """
    Compute Mel Frequency Cepstral Coefficients.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :param coeffs: number of coefficients
    :param filters_count: number of filters
    :param fft_size: n-points of discrete fourier transform
    :param low_freq: start frequency of first filter
    :param high_freq: end frequency of last filter
    :return: numpy array with mel frequency cepstral coefficients
    """
    log_filterbank_energies = get_filterbank_energies(signal, rate, frame_length, frame_step, window_function,
                                                      filters_count, fft_size,
                                                      low_freq, high_freq, True)
    mfcc = dct(log_filterbank_energies, type=2, axis=1, norm='ortho')[:, :coeffs]

    return mfcc


def get_deltas(matrix, axis=0, order=1):
    """
    Compute n-th order deltas.
    :param matrix: matrix with features
    :param axis: axis to calculate difference along
    :param order: order of discrete difference
    :return: numpy array with deltas
    """
    deltas = np.diff(matrix, n=order, axis=axis)

    # add padding of zeroes -> before: order x 0, after 0 x 0
    padding = [(0, 0)] * matrix.ndim
    padding[axis] = (order, 0)
    deltas = np.pad(deltas, padding, mode='constant')

    return deltas
