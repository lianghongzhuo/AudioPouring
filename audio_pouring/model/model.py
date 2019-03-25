#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name     : model.py
# Purpose       :
# Creation Date : 05-12-2018
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn


class HeightRegression(nn.Module):
    """ height regression from multi data embedding space"""

    def __init__(self, input_size=128, hidden_size=64, layer_num=1, output_size=1):
        super(HeightRegression, self).__init__()
        self.layer_num = layer_num
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

        self.drop = nn.Dropout(0.5)
        if layer_num == 2:
            self.reg = nn.Linear(hidden_size // 2, output_size)
        elif layer_num == 1:
            self.reg = nn.Linear(hidden_size, output_size)
        else:
            exit("In class HeightRegression: No such layer number")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        if self.layer_num == 2:
            x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.reg(x))
        return x


class ScaleRegression(nn.Module):
    """ scale regression from multi data embedding space"""

    def __init__(self, batch_size, input_size=128, output_size=1, is_sum=False):
        super(ScaleRegression, self).__init__()

        if is_sum:
            hidden_size = 32
        else:
            hidden_size = 16
        if input_size == 1:
            hidden_size = 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.reg = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.is_sum = is_sum
        self.batch_size = batch_size

    def forward(self, x):
        if self.is_sum:
            x = self.relu(self.bn1(self.fc1(x))).view(self.batch_size, -1)
            x = torch.cumsum(x, dim=1).view(-1, 1)
        else:
            x = self.relu(self.bn1(self.fc1(x)))
            # x = self.reg(x)
        return x


class AudioRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, mini_batch_size, num_layers=1, audio_fc1_size=200, dropout=0.5,
                 bidirectional=False, is_lstm=True):
        super(AudioRNN, self).__init__()
        self.num_layers = num_layers  # rnn layer num
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.is_lstm = is_lstm

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mini_batch_size = mini_batch_size
        self.hidden = self.init_hidden()
        self.audio_fc1_size = audio_fc1_size

        if is_lstm:
            self.rnn = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                               bidirectional=self.bidirectional, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional, batch_first=True)

        fc_layer_num = 1
        self.height = HeightRegression(input_size=self.hidden_dim, layer_num=fc_layer_num)

        # initialize bias
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1)

    def init_hidden(self, device=None):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        num_direction = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_direction, self.mini_batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers * num_direction, self.mini_batch_size, self.hidden_dim, device=device)
        if self.is_lstm:
            return h0, c0
        else:
            return h0

    def forward(self, x):
        """
        :param x: [0: batch size 1: sequence length 2: input size]
        :return:
        """
        # x = self.relu(self.bn1(self.fc1(x.reshape(-1, self.input_dim))))
        # y = x.view(-1, self.mini_batch_size, self.input_dim)  # stupid error remember forever
        assert len(x.size()) == 3, '[RNN]: Input dimension must be of length 3'
        assert self.mini_batch_size == x.size()[0], "[RNN]: Input mini batch size must equal to the input data size"
        rnn_out, self.hidden = self.rnn(x, self.hidden)
        height = self.height(rnn_out.reshape(-1, self.hidden_dim))
        return height, self.hidden


class AudioCNNRealT(nn.Module):
    def __init__(self, feature_size=257, hidden_size=128):
        super(AudioCNNRealT, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn4 = nn.BatchNorm1d(hidden_size // 4)
        self.drop4 = nn.Dropout(0.5)
        self.height = nn.Linear(hidden_size // 4, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.pool0(self.relu(self.conv0(x)))
        # x = self.pool1(self.relu(self.conv1(x))).view(-1, self.feature_size)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        # x = self.relu(self.bn4(self.fc4(x)))
        x = self.height(x)

        return x


if __name__ == '__main__':
    # test code for AudioLSTM:
    seq_length = 520
    mini_batch_size_test = 2
    input_size_test = 225
    hidden_dim_test = 50
    x_input_tmp = torch.ones(mini_batch_size_test, seq_length, input_size_test)
    model = AudioRNN(input_size_test, hidden_dim_test, mini_batch_size_test)
    output_tmp, hidden = model(x_input_tmp)
    print(output_tmp.shape)
