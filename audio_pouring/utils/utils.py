#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name     :
# Purpose       :
# Creation Date : 05-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]

from __future__ import print_function, division
import numpy as np
import glob
import os
import sys
import yaml
import pickle
from scipy import interpolate
import librosa.util
import librosa.display
import audio_pouring

config_file = os.path.join(os.path.dirname(audio_pouring.__file__), "config/preprocess.yaml")
config = yaml.load(open(config_file, "r"))


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


def get_file_name(file_dir_):
    file_list = []
    root_list = []
    for root, dirs, files in os.walk(file_dir_):
        # if root.count("/") == file_dir_.count("/")+1:
        file_list += [file_ for file_ in files]
        if root.count("/") == file_dir_.count("/") + 1:
            root_list.append(root)
    file_list.sort()
    return file_list, root_list


def generate_npy(npy_path_):
    train0 = np.array([])
    test0 = np.array([])

    for i in range(2, len(sys.argv)):
        bottle = sys.argv[i]
        files_train = glob.glob(os.path.join(npy_path_ + str(bottle) + "_train", "*.npy"))
        files_test = glob.glob(os.path.join(npy_path_ + str(bottle) + "_test", "*.npy"))

        train = np.array(files_train)
        test = np.array(files_test)

        # d_num = []
        # for num in range(train.shape[0] - 20, train.shape[0]):
        #     if test[0].split("/")[-1][:21] == train[num].split("/")[-1][:21]:
        #         d_num += [num]
        # train = np.delete(train, d_num, axis=0)

        np.save(npy_path_ + str(bottle) + "_train.npy", train)
        np.save(npy_path_ + str(bottle) + "_test.npy", test)
        train0 = np.hstack([train0, train])
        test0 = np.hstack([test0, test])

    np.save(npy_path_ + str(0) + "_train.npy", train0)
    np.save(npy_path_ + str(0) + "_test.npy", test0)
    print("All finish")


def audio_first_normalization(pickle_path):
    # pickle path should be at data/bag/pickle
    target_frequency = config["frequency"]
    orig_sr = config["source_frequency"]
    pickle1_files = glob.glob(os.path.join(pickle_path + str(1), "*.pickle"))
    pickle3_files = glob.glob(os.path.join(pickle_path + str(3), "*.pickle"))
    pickle4_files = glob.glob(os.path.join(pickle_path + str(4), "*.pickle"))
    pickle_files = pickle1_files + pickle3_files + pickle4_files
    pickle_num = len(pickle_files)

    # delete bad data
    delete_name = np.load("../del.npy")
    n = 0
    max_audio = -100
    for i in range(pickle_num):
        data = []
        with (open(pickle_files[i], "rb")) as openfile:
            data.append(pickle.load(openfile))
            if pickle_files[i].split("/")[-1][:-7] in delete_name:
                n = n + 1
                print(n)
                print("delete bad bag ", pickle_files[i].split("/")[-1][:-7])
            else:
                audio_raw = data[0]["audio"]
                audio_resample = librosa.core.resample(audio_raw, orig_sr=orig_sr, target_sr=target_frequency)
                max_audio = max(np.max(audio_resample), max_audio)

    print(max_audio)


def normalization_all_data(npy_path_):
    npy1_files = glob.glob(os.path.join(npy_path_ + str(1) + "_train", "*.npy"))
    npy1_files_test = glob.glob(os.path.join(npy_path_ + str(1) + "_test", "*.npy"))
    npy3_files = glob.glob(os.path.join(npy_path_ + str(3) + "_train", "*.npy"))
    npy3_files_test = glob.glob(os.path.join(npy_path_ + str(3) + "_test", "*.npy"))
    npy4_files = glob.glob(os.path.join(npy_path_ + str(4) + "_train", "*.npy"))
    npy4_files_test = glob.glob(os.path.join(npy_path_ + str(4) + "_test", "*.npy"))
    files = npy1_files + npy3_files + npy4_files + npy1_files_test + npy3_files_test + npy4_files_test
    files.sort()
    print("data num :", len(files))
    nn = []
    for file_ in files:
        nn.append(np.load(file_))
    d = np.array(nn)
    audio = np.vstack(d[:, 1])
    audio_mean = np.mean(audio)
    audio_std = np.std(audio)
    print("audio_mean", audio_mean)
    print("audio_std", audio_std)

    height = np.vstack(d[:, 3])
    print("cavity h max :", np.max(height))
    print("cavity h min :", np.min(height))

    scale = np.vstack(d[:, 4])
    print("scale change max :", np.max(scale))
    print("scale change min :", np.min(scale))

    return audio_mean, audio_std


def weight2height(cup_id, cur_weight):
    """
    function to get "real" water height from weight
    :param cup_id: cup id in str
    :param cur_weight: input weight in kg
    :return: cavity_h: cavity height in mm
    """
    if cup_id in ["1", "3", "4", "6", "7", "8"]:
        # params from bottle_config
        bottle_config_path = os.path.join(get_pouring_path(), "config/bottles/bottle" + cup_id + "_config.npy")
        params = np.load(bottle_config_path)
        poly = np.polynomial.Polynomial(params)
        cavity_h = poly(cur_weight)
    else:
        print("wrong type of cups")
        cavity_h = "WRONG CUP_ID INPUT"

    return cavity_h


def height2weight(cup_id, height):
    out = 200
    i = 1
    while out > height:
        out = weight2height(cup_id, cur_weight=i * 0.0001)
        i += 1
    print(out, i * 0.0001)
    return i * 0.0001


def get_pouring_path():
    return os.path.dirname(audio_pouring.__file__)


if __name__ == "__main__":
    print("put one or more function to use")
    npy_path = os.path.join(get_pouring_path(), "dataset")
    # generate_npy(npy_path)
    # audio_first_normalization(pickle_path)  # please specify pickle path here
    # normalization_all_data()
