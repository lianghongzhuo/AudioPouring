#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang
# E-mail     : liang@informatik.uni-hamburg.de
# Description:
# Date       : 20/03/2019: 4:36 PM
# File Name  : generate_npy_list
from __future__ import print_function
import glob
import os
from audio_pouring.utils.utils import get_pouring_path
import numpy as np

if __name__ == "__main__":
    dataset_path = os.path.join(get_pouring_path(), "dataset")
    dir_list = os.listdir(dataset_path)
    for npy_dir in dir_list:
        full_path = os.path.join(dataset_path, npy_dir)
        if os.path.isdir(full_path):
            npy_names = glob.glob(os.path.join(full_path, "*.npy"))
            if len(npy_names) != 0:
                print("save npy list in {} to a npy file".format(npy_dir))
                np.save(full_path + ".npy", npy_names)

    # combine generated npy files as bottle 0:
    npy_train_list_names = glob.glob(os.path.join(dataset_path, "npy_train*.npy"))
    npy_test_list_names = glob.glob(os.path.join(dataset_path, "npy_test*.npy"))
    bottle0_train = np.array([])
    bottle0_test = np.array([])
    for list_file in npy_train_list_names:
        if not list_file.split("/")[-1] == "npy_train_0.npy":
            bottle0_train = np.hstack([bottle0_train, np.load(list_file)])
    for list_file in npy_test_list_names:
        if not list_file.split("/")[-1] == "npy_test_0.npy":
            bottle0_test = np.hstack([bottle0_test, np.load(list_file)])
    np.save(os.path.join(dataset_path, "npy_train_0.npy"), bottle0_train)
    np.save(os.path.join(dataset_path, "npy_test_0.npy"), bottle0_test)
    print("save bottle 0 data")
