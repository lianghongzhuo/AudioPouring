#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name     : dataset.py
# Purpose       :
# Creation Date : 05-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]
from __future__ import division, print_function
import os
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as trans
import numpy as np
import yaml
from audio_pouring.utils.utils import get_pouring_path

config_file = os.path.join(get_pouring_path(), "config/preprocess.yaml")
config = yaml.load(open(config_file, "r"))


def pitch_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols // 20  # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=0)


class PouringDataset(torch.utils.data.Dataset):
    def __init__(self, path, input_audio_size, multi_modal=False, train_rnn=False, is_fixed=False,
                 is_train=False, is_filtered=False, is_noise=False, bottle_train='1', bottle_test='1', seg_audio=False):
        self.path = path
        self.input_audio_size = input_audio_size

        self.is_train = is_train
        self.multi_modal = multi_modal
        self.train_rnn = train_rnn
        self.bottle_train = bottle_train
        self.bottle_test = bottle_test
        self.seg_audio = seg_audio
        # self.seq_length = seq_length

        if is_filtered:
            self.audio_mean = config["filtered"]["audio_mean"]
            self.audio_std = config["filtered"]["audio_std"]
        else:
            self.audio_mean = config["unfiltered"]["audio_mean"]
            self.audio_std = config["unfiltered"]["audio_std"]

        if is_train:
            if self.train_rnn:
                if is_noise:
                    self.label = np.load(path + 'noise_good_fixed_bottle' + self.bottle_train + '_train.npy')
                else:
                    if is_filtered:
                        self.label = np.load(path + 'filter_npy' + self.bottle_train + '_train.npy')
                    else:
                        if is_fixed:
                            self.label = np.load(path + 'robot_train0.npy')
                        else:
                            self.label = np.load(path + 'bottle' + self.bottle_train + '_train.npy')
            else:
                self.label = np.load(path + 'bottle' + self.bottle_train + '_train.npy')
        else:  # load test dataset
            if self.train_rnn:
                if is_noise:
                    self.label = np.load(path + 'good_fixed_bottle' + self.bottle_test + '_test.npy')
                else:
                    if is_filtered:
                        self.label = np.load(path + 'filter_npy' + self.bottle_train + '_test.npy')
                    else:
                        if is_fixed:
                            self.label = np.load(path + 'robot_test0.npy')
                        else:
                            self.label = np.load(path + 'bottle' + self.bottle_test + '_test.npy')
            else:
                self.label = np.load(path + 'bottle' + self.bottle_test + '_test.npy')

        self.length = len(self.label)

    def __getitem__(self, index):
        tmp_path = self.label[index].split("/")[-2:]
        tag = np.load(os.path.join(self.path, tmp_path[0], tmp_path[1]))
        if not self.train_rnn:
            tag = np.squeeze(tag)
            target = np.array(tag[2]).astype(np.float32)
        else:
            target = np.array(tag[3]).astype(np.float32)
        audio = tag[1].astype(np.float32)
        audio -= self.audio_mean
        audio /= self.audio_std
        if self.seg_audio:
            audio = audio[20:120]

        assert (audio.shape[0] == self.input_audio_size)

        #   Augmented(if train/is_noise)
        #   if self.is_train:
        #       audio = pitch_shift_spectrogram(audio)

        if self.train_rnn:
            assert (target.shape[0] == audio.shape[1])
            return audio.T, target
        else:
            return audio, target

    def __len__(self):
        return self.length


if __name__ == "__main__":
    b = PouringDataset("../dataset/", input_audio_size=257, input_force_size=1, multi_modal=False,
                       train_rnn=True, is_fixed=True, is_train=True, seg_audio=False)
    train_loader = torch.utils.data.DataLoader(b, batch_size=1, num_workers=32, pin_memory=True, )
    for batch_idx, (audio_, height_) in enumerate(train_loader):
        print(batch_idx, audio_.shape, height_.shape)
    a, f, h, s = b.__getitem__(1)
