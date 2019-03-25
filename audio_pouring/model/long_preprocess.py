#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name     : data_preprocess.py
# Purpose       :
# Creation Date : 03-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]com]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]com]
from __future__ import division
from __future__ import print_function
import numpy as np
import glob
import os
import sys
import pickle
import math
import yaml
import pylab
import matplotlib.pyplot as plt
import librosa.util
import librosa.display
from scipy.signal import butter, filtfilt
from scipy import interpolate
import multiprocessing as mp
from audio_pouring.utils.utils import weight2height, get_pouring_path
import seaborn as sns

pkg_path = get_pouring_path()
config_file = os.path.join(pkg_path, "config/preprocess.yaml")
config = yaml.load(open(config_file, "r"))
fixed_length = config["fixed_audio_length"]  # unit seconds
target_frequency = config["frequency"]
n_fft = config["n_fft"]
win_size = config["win_size"]
overlap = config["overlap"]
win_length = int(win_size * target_frequency)  # win length, how many samples in this window
assert win_length >= n_fft, "win_length should larger equal than n_fft"
hop_length = int(win_length * overlap)
max_audio = config["max_audio"]
bottle_id_list = config["bottle_id_list"]

same_length_mode = True  # random sample a fixed length of sample
is_robot = False  # pouring data from robot or human
# data augmentation
add_noise = False
add_filter = False
# vis
vis_audio = False
vis_ros_real_time = False
vis_scale = False
np.random.seed(1)


class Interp1dPickleAble:
    """ class wrapper for piecewise linear function
    """

    def __init__(self, xi, yi, **kwargs):
        self.xi = xi
        self.yi = yi
        self.args = kwargs
        self.f = interpolate.interp1d(xi, yi, **kwargs)

    def __call__(self, xnew):
        return self.f(xnew)

    def __getstate__(self):
        return self.xi, self.yi, self.args

    def __setstate__(self, state):
        self.f = interpolate.interp1d(state[0], state[1], **state[2])


def main(pickle_path, npy_path, multi_threading):
    pickle_files = glob.glob(os.path.join(pickle_path, "*.pickle"))
    pickle_num = len(pickle_files)

    if multi_threading:
        num_workers = mp.cpu_count()

        def task(task_inter_, i_, pickle_files_, npy_path_):
            i_ = i_ + task_inter_ * num_workers
            pickle_file_ = pickle_files_[i_]
            # print("Working on:", pickle_file_)
            data_ = pickle.load(open(pickle_file_, "rb"))
            npy_generate(data_, pickle_file_, npy_path_)

        num_inter = int(math.ceil(pickle_num / num_workers))
        last_tasks_num = pickle_num % num_workers
        for task_round_inter_ in range(num_inter):
            if task_round_inter_ == num_inter - 1 and last_tasks_num > 0:
                n = last_tasks_num
            else:
                n = num_workers
            workers = [mp.Process(target=task,
                                  args=(task_round_inter_, i, pickle_files, npy_path)) for i in range(n)]
            [i.start() for i in workers]
    else:
        for i in range(pickle_num):
            with (open(pickle_files[i], "rb")) as openfile:
                data = pickle.load(openfile)
                npy_generate(data, pickle_files[i], npy_path)


def check_audio_time(audio_raw, audio_time, orig_sr, pickle_name):
    time_audio_ros = audio_time[-1] - audio_time[0]
    time_audio = audio_raw.shape[0] / orig_sr
    tolerance_audio_tootle_time = 0.05
    tolerance_audio_message_time = 0.1
    audio_diff = audio_time[1:] - audio_time[:-1]
    if np.where(audio_diff > tolerance_audio_message_time)[0].shape[0] > 0:
        print("[ERROR] audio message at {} has abnormal message time gaps".format(pickle_name))
        return False
    elif abs(time_audio - time_audio_ros) > tolerance_audio_tootle_time:
        print("[ERROR] audio message at {} is not match time audio ros time".format(pickle_name))
        return False
    else:
        return True


def npy_generate(data, pickle_file, npy_path):
    # get the clip num based on the real audio time
    pickle_name = pickle_file.split("/")[-1]
    audio_raw = data["audio"]
    frame_size = data["frame_size"]
    audio_time = data["audio_time"]
    scale_time = data["time"]
    wrench = data["wrench"]
    wrench_size = config["wrench_size"]
    wrench_time = data["wrench_time"]
    f_scale = data["f_scale"]
    orig_sr = data["sample_frequency"]
    audio_time_length = audio_time[-1] - audio_time[0]

    if not check_audio_time(audio_raw, audio_time, orig_sr, pickle_name):
        return  # if the audio is not correct, return directly
    # based on the audio length generate augmentation numbers
    aug_numbers = int(math.ceil((audio_time_length - fixed_length) * 2) / 2 * 10)

    # Floating point precision
    fp = int(str(int(audio_time[0]))[:-2]) * math.pow(10, 2)
    audio_time = audio_time - fp
    ros_time = np.unique(audio_time)
    ros_time_delta = (ros_time[1:] - ros_time[:-1]).reshape(-1, 1)
    ind_good = np.where(ros_time_delta > 1e-3)[0]  # some messages have close time stamp

    if ind_good[0] != 0:
        ind_good = np.hstack([0, ind_good])
    if ind_good[-1] != len(ros_time_delta):
        ind_good = np.hstack([ind_good, len(ros_time_delta)])

    ros_real_time = np.array([])
    for i in range(1, len(ind_good)):
        tmp_ind_diff = ind_good[i] - ind_good[i - 1]
        tmp_ranges = np.array(range(frame_size * tmp_ind_diff - 1, -1, -1))
        tmp_time_diff = ros_time[ind_good[i]] - ros_time[ind_good[i - 1]]
        tmp = ros_time[ind_good[i]] - tmp_ranges * tmp_time_diff / tmp_ind_diff / frame_size
        ros_real_time = np.hstack([ros_real_time, tmp])
    if vis_ros_real_time:
        sns.set(palette="deep", color_codes=True)
        with sns.axes_style("darkgrid"):
            plt.title("real-time per audio data", fontsize=20)
            plt.plot(ros_real_time, "r.", label="estimated time points")
            plt.plot(audio_time[frame_size:], "b-", label="ROS time stamp")
            plt.plot(np.array(range(len(ros_time))) * frame_size, ros_time, "*", label="message")
            plt.plot(ind_good * frame_size, ros_time[ind_good], "go", label="connection points")
            plt.legend(loc=2)
            plt.show()

    wrench_time_single = wrench_time.reshape(-1, wrench_size)[:, 0] - fp
    f_magnitude, f_raw = wrench_process(wrench, wrench_time, False, pickle_name[0:-7])

    if same_length_mode:
        start_max = max(0, int(len(audio_raw) - orig_sr * fixed_length) - 1)
        if start_max < aug_numbers:
            aug_numbers = start_max + 1
        start_index = np.random.choice(range(0, start_max + 1), aug_numbers, replace=False)
        end_index = start_index + fixed_length * orig_sr
    else:
        start_index = np.array([0])
        end_index = None

    for ind in range(len(start_index)):
        s_index = start_index[ind]
        e_index = end_index[ind]
        cavity_h = np.array([])
        whole_scale = np.array([])

        audio_spec = audio_process(audio_raw[s_index: e_index], orig_sr, target_frequency, vis_audio,
                                   pickle_name[0:-7] + "_" + str(ind))

        # resampled real column corresponding to each column in spectrogram
        sample_half_column = win_size * orig_sr / 2
        column_end_index = (sample_half_column * np.array(range(1, audio_spec.shape[1] + 1)) + s_index).astype(int)
        column_start_index = np.hstack([int(column_end_index[0] - sample_half_column), column_end_index[:-1]])
        column_end_index = column_end_index[1:-2]
        column_start_index = column_start_index[1:-2]
        audio_spec = audio_spec[:, 1:-2]
        for i, j in zip(column_start_index, column_end_index):
            c_start_time = ros_real_time[i]
            c_end_time = ros_real_time[j]

            # get corresponding weight and cavity height
            current_scale = f_scale(c_end_time - (scale_time[0] - fp))
            if current_scale < 0:
                print("there is a minor mismatch in ros time, {}".format(current_scale))
                print("please check your data if this message is too often")
                current_scale = 0
            cur_cavity_h = weight2height(pickle_name[0], current_scale)

            cavity_h = np.hstack([cavity_h, cur_cavity_h])
            whole_scale = np.hstack([whole_scale, current_scale])

        # relative scale
        whole_scale = whole_scale * 1000
        whole_scale_relative = whole_scale - whole_scale[0]

        if vis_scale:
            sns.set(palette="deep", color_codes=True)
            with sns.axes_style("darkgrid"):
                y = data["scale"] * 1000
                s_time = data["time"]
                x = s_time - s_time[0]
                whole_time = ros_real_time[column_end_index] - s_time[0] + fp
                x_new = np.linspace(x[0], x[-1], 50)
                plt.title("scale/weight curve", fontsize=20)
                plt.plot(x, y, "ro", label="raw scale")
                plt.plot(x_new, f_scale(x_new) * 1000, "b-", label="interpolated scale")
                plt.plot(whole_time, whole_scale, "g--", label="current weight")
                plt.plot(whole_time, cavity_h, "b--", label="current air column height")

                # if plot cavity and scale together, need to command xlim line
                plt.xlim([x[0] - 1, x[-1] + 1])
                plt.xlabel("time (second)", fontsize=16)
                plt.ylabel("Weight (g)", fontsize=16)
                plt.grid()
                plt.legend(loc=2)
                plt.show()

        # save data to *.npy
        name = pickle_name[:-7] + "-" + str(ind)
        npy_name = os.path.join(npy_path, name + ".npy")
        # print("save data to ", name)
        np.save(npy_name, [np.array([name]), audio_spec, cavity_h, whole_scale_relative])


def audio_band_filter(audio_raw):
    fl = 0.005
    fh = 0.8
    b = 0.05
    nn = int(np.ceil((4 / b)))
    if not nn % 2:  # Make sure that N is odd.
        nn += 1
    n = np.arange(nn)

    # low-pass filter
    hlpf = np.sinc(2 * fh * (n - (nn - 1) / 2.))
    hlpf *= np.blackman(nn)
    hlpf = hlpf / np.sum(hlpf)

    # high-pass filter
    hhpf = np.sinc(2 * fl * (n - (nn - 1) / 2.))
    hhpf *= np.blackman(nn)
    hhpf = hhpf / np.sum(hhpf)
    hhpf = 0 - hhpf
    hhpf[int((nn - 1) / 2)] += 1
    h = np.convolve(hlpf, hhpf)
    s = list(audio_raw)
    new_signal = np.convolve(s, h)
    return new_signal


def audio_high_filter(audio_raw):
    fc = 0.01
    b = 0.01
    nn = int(np.ceil((4 / b)))
    if not nn % 2:
        nn += 1
    n = np.arange(nn)

    sinc_func = np.sinc(2 * fc * (n - (nn - 1) / 2.))
    window = np.blackman(nn)
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    # reverse function
    sinc_func = 0 - sinc_func
    sinc_func[int((nn - 1) / 2)] += 1
    s = list(audio_raw)
    new_signal = np.convolve(s, sinc_func)
    return new_signal


def audio_process(audio, orig_sr, target_frequency_, vis, fig_name):
    audio_resample = librosa.core.resample(audio, orig_sr=orig_sr, target_sr=target_frequency_)
    audio_resample *= 1. / max_audio * 0.9
    if add_noise:
        wn = np.random.randn(len(audio_resample))
        dampening_factor = 0.0002
        audio_resample = audio_resample + dampening_factor * wn
    if add_filter:
        band_filter = False
        if band_filter:
            audio_resample = audio_band_filter(audio_resample)
        else:
            audio_resample = audio_high_filter(audio_resample)
    audio_re_fft = librosa.stft(y=audio_resample, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    audio_re_fft = np.abs(audio_re_fft)
    audio_re_db = librosa.core.amplitude_to_db(audio_re_fft)

    if vis:
        audio_fft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        audio_fft = np.abs(audio_fft)
        audio_db = librosa.core.amplitude_to_db(audio_fft)

        fig = pylab.gcf()
        fig.canvas.set_window_title(fig_name)
        fig.set_size_inches(8., 3)
        plt.subplot(1, 2, 1)
        librosa.display.specshow(audio_db, sr=orig_sr, hop_length=hop_length, y_axis="linear", x_axis="time")
        plt.title("Original Spectrogram")
        plt.tight_layout()
        plt.colorbar()
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.subplot(1, 2, 2)
        librosa.display.specshow(audio_re_db, sr=target_frequency_, hop_length=hop_length, y_axis="linear",
                                 x_axis="time")
        plt.title("Resampled Spectrogram")
        plt.tight_layout()
        plt.colorbar()
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.yticks(np.arange(0, 8000.01, 2000), fontsize=12)
        plt.xticks(fontsize=12)
        plt.show()
    return audio_re_db


def butter_low_pass_filter(data, cutoff, fs, order):
    """
    :param data   : input data
    :param cutoff : desired cutoff frequency of the filter, Hz
    :param fs     : sample rate, Hz
    :param order  :
    :return       : filtered data
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    [b, a] = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


if __name__ == "__main__":
    usage = "usage: python long_preprocess.py [train/test] [mt/test]"
    if len(sys.argv) != 3:
        exit(usage)
    mode = sys.argv[1]  # choice train, test
    mt = False
    if sys.argv[2] == "mt":
        mt = True
    elif sys.argv[2] == "test":
        mt = False
    else:
        exit(usage)
    for bottle in bottle_id_list[:3]:
        if mode == "train":  # process training data
            save_npy_path = os.path.join(pkg_path, "dataset/npy_train_" + str(bottle))
            source_pickle_path = os.path.join(pkg_path, "pickle/pickle_train_" + str(bottle))
        elif mode == "test":  # process test data
            save_npy_path = os.path.join(pkg_path, "dataset/npy_test_" + str(bottle))
            source_pickle_path = os.path.join(pkg_path, "pickle/pickle_test_" + str(bottle))
        else:
            raise KeyError("no such mode: {}".format(mode))
        if not os.path.isdir(save_npy_path):
            os.mkdir(save_npy_path)
        main(source_pickle_path, save_npy_path, multi_threading=mt)
