import pywt
import scipy
import numpy as np
import scipy.fftpack
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def nomrmalize(data):
    """
    Perform min-max normalization on the data
    :param data:
    :return:
    """
    return (data - np.min(data))/(np.max(data)-np.min(data))


def uint8_to_float(img):
    """
    Convert an image from uint8 to float
    :param img:
    :return:
    """
    result = np.ndarray(shape=img.shape, dtype='float')
    result[:] = img * (1. / 255)
    return result


def float_to_uint8(img):
    """
    Convert an image from float to uint8
    :param img:
    :return:
    """
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = img * 255
    return result


def float_to_int8(img):
    """
    Convert an image from float to int8
    :param img:
    :return:
    """
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = (img * 255) - 127
    return result


def butter_bandpass(_lowcut, _highcut, _fs, order=5):
    """
    Create a Butterworth bandpass filter
    :param _lowcut:
    :param _highcut:
    :param _fs:
    :param order:
    :return:
    """
    _nyq = 0.5 * _fs
    _low = _lowcut / _nyq
    _high = _highcut / _nyq
    # noinspection PyTupleAssignmentBalance
    _b, _a = scipy.signal.butter(order, [_low, _high], btype='band', output='ba')
    return _b, _a


def butter_bandpass_filter(_data, _lowcut, _highcut, _fs, order=5):
    """
    Apply a Butterworth bandpass filter to the data
    :param _data:
    :param _lowcut:
    :param _highcut:
    :param _fs:
    :param order:
    :return:
    """
    _b, _a = butter_bandpass(_lowcut, _highcut, _fs, order=order)
    _y = scipy.signal.lfilter(_b, _a, _data)
    return _y


def butter_bandpass_filter_fast(_data, _b, _a, axis=0):
    """
    Apply a Butterworth bandpass filter to the data
    :param _data:
    :param _b:
    :param _a:
    :param axis:
    :return:
    """
    _y = scipy.signal.lfilter(_b, _a, _data, axis=axis)
    return _y


def butter_lowpass(cutoff, fs, order=5):
    """
    Create a Butterworth lowpass filter
    :param cutoff:
    :param fs:
    :param order:
    :return:
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth lowpass filter to the data
    :param data:
    :param cutoff:
    :param fs:
    :param order:
    :return:
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0,
                             amplification_factor=50, verbose=False, debug=''):
    """
    Apply a temporal bandpass filter to the data
    :param data:
    :param fps:
    :param freq_min:
    :param freq_max:
    :param axis:
    :param amplification_factor:
    :param verbose:
    :param debug:
    :return:
    """
    b, a = butter_bandpass(freq_min, freq_max, fps, order=6)
    result = butter_bandpass_filter_fast(data, b, a, axis=axis)
    result *= amplification_factor
    if verbose:
        print('{0}{1},{2}'.format(debug, result.min(), result.max()))
    return result


def temporal_bandpass_filter_fft(data, fps, freq_min=0.833, freq_max=1, axis=0,
                                 amplification_factor=50, verbose=False, debug=''):
    """
    Apply a temporal bandpass filter to the data using FFT
    :param data:
    :param fps:
    :param freq_min:
    :param freq_max:
    :param axis:
    :param amplification_factor:
    :param verbose:
    :param debug:
    :return:
    """
    data_shape = (len(data), data[0].shape[0], data[0].shape[1])
    # noinspection PyUnresolvedReferences
    fft = scipy.fftpack.rfft(data, axis=axis)
    # noinspection PyUnresolvedReferences
    frequencies = scipy.fftpack.fftfreq(data_shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[bound_high:-bound_high] = 0
    if bound_low != 0:
        fft[:bound_low] = 0
        fft[-bound_low:] = 0

    result = np.ndarray(shape=data_shape, dtype='float')
    # noinspection PyUnresolvedReferences
    result[:] = np.real(scipy.fftpack.ifft(fft, axis=0))
    result *= amplification_factor
    if verbose:
        print('{0}{1},{2}'.format(debug, result.min(), result.max()))
    return result

def wavelet_filter(data, w='db4', iterations=5):
    """
    Perform wavelet filtering on the data
    :param data:
    :param w:
    :param iterations:
    :return:
    """
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(iterations):
        (a, d) = pywt.dwt(a, w, pywt.Modes.smooth)
        ca.append(a)
        cd.append(d)

    rec_a = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))
    return rec_a[-1]
