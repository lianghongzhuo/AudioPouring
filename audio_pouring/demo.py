#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name     : main-lstm.py
# Purpose       :
# Creation Date : 05-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]
from __future__ import division, print_function
import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as torch_func
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from model.dataset import PouringDataset
from model.model import AudioRNN
import librosa.util
import glob
import rospy
from geometry_msgs.msg import WrenchStamped
from portaudio_transport.msg import AudioTransport
from std_msgs.msg import Bool
from audio_pouring.utils.utils import weight2height, get_pouring_path

parser = argparse.ArgumentParser(description="audio2height")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--bottle", type=str, choices=["1", "3", "4", "6", "7", "8"], required=True)
parser.add_argument("--cavity-height", type=float, default=20.0)
parser.add_argument("--model-path", type=str,
                    default="./assets/learned_models/robot_experiment/",
                    help="pre-trained model path")
parser.add_argument("--filter", action="store_true")

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available() else False

if args.cuda:
    torch.cuda.manual_seed(1)

np.random.seed(int(time.time()))
config_file = os.path.join(get_pouring_path(), "config/preprocess.yaml")
config = yaml.load(open(config_file, "r"))
target_frequency = config["frequency"]
source_frequency = config["source_frequency"]  # unit Hz
audio_topic = config["audio_topic"]
ft_topic = config["ur5_ft_topic"]
audio_length = config["fixed_audio_length"]  # unit second
n_fft = config["n_fft"]
win_size = config["win_size"]
overlap = config["overlap"]
win_length = int(win_size * target_frequency)
hop_length = int(win_length * overlap)
x_bias = config["ur5_x"]
y_bias = config["ur5_y"]
z_bias = config["ur5_z"]
thresh_acc = np.array([1, 2, 3, 4, 5])  # unit mm

if args.filter:
    audio_mean = config["filtered"]["audio_mean"]
    audio_std = config["filtered"]["audio_std"]
else:
    audio_mean = config["unfiltered"]["audio_mean"]
    audio_std = config["unfiltered"]["audio_std"]
input_audio_size = config["input_audio_size"]
bottle_upper = torch.tensor([config["bottle_upper"]])
bottle_lower = torch.tensor([config["bottle_lower"]])
max_audio = config["max_audio"]

model_path = args.model_path + "lstm_bottle0.model"

if args.cuda:
    model = torch.load(model_path, map_location="cuda:{}".format(args.gpu))
    model.device_ids = [args.gpu]
else:
    model = torch.load(model_path, map_location="cpu")
model.mini_batch_size = 1

device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0, 1]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    bottle_upper = bottle_upper.cuda()
    bottle_lower = bottle_lower.cuda()


def test(model_, audio):
    model_.eval()
    torch.set_grad_enabled(False)
    if args.cuda:
        audio = audio.cuda()
    model_.hidden = model_.init_hidden(device)
    height, hidden = model_(audio)
    height = height * (bottle_upper - bottle_lower) + bottle_lower
    return height.cpu().data.numpy()


class Pouring:
    def __init__(self, target_cavity_height):
        self.target = target_cavity_height
        self.audio_numpy = np.array([])
        self.last_height = 0
        self.scale_reading = 0.0
        self.scale_reading_last = 0.0
        self.not_print = True
        self.pub = rospy.Publisher("stop_pour", Bool, queue_size=10)
        self.scale_sub = rospy.Subscriber("/maul_logic/wrench", WrenchStamped, self.scale_callback)
        rospy.sleep(2)  # wait the scale to have reading
        self.pouring_run()
        rospy.spin()

    def scale_callback(self, scale_data):
        self.scale_reading_last = self.scale_reading
        self.scale_reading = scale_data.wrench.force.z

    def pouring_run(self):
        while self.scale_reading == self.scale_reading_last:
            pass
        rospy.Subscriber(audio_topic, AudioTransport, self.audio_callback)
        rospy.sleep(0.5)
        show_net_work_time = False
        while not rospy.is_shutdown():
            start_time = time.time()
            audio_spectrum = torch.Tensor(self.audio_process(self.audio_numpy).T)
            if show_net_work_time:
                print("spectrum process time:", time.time() - start_time)
            start_time = time.time()
            audio_spectrum = audio_spectrum.unsqueeze(0)
            if show_net_work_time:
                print("network process time:", time.time() - start_time)
            all_height = test(model, audio_spectrum)
            self.process_height(all_height)

    def process_height(self, all_height):
        height = all_height[-1][0] - 5
        print("Current length of the air column is {}, go on pouring".format(height))
        if height <= self.target and abs(height - self.last_height) < 8:
            rospy.loginfo("Enjoy your drink!")
            start_time = time.time()
            while not rospy.is_shutdown():
                self.pub.publish(True)
                real_height = weight2height(cup_id=args.bottle, cur_weight=self.scale_reading)
                if time.time() - start_time > 4 and self.not_print:
                    rospy.loginfo("Network {}mm, Real {}mm, Desire {}mm, Scale {}kg".format(height, real_height,
                                                                                            self.target,
                                                                                            self.scale_reading))
                    self.not_print = False
        self.last_height = height

    def audio_callback(self, audio_raw):
        audio_numpy = np.array(audio_raw.channels[0].frame_data)
        self.audio_numpy = np.hstack([self.audio_numpy, audio_numpy])
        # get latest 4 second messages
        if self.audio_numpy.shape[0] > source_frequency * audio_length:
            self.audio_numpy = self.audio_numpy[-source_frequency * audio_length:]

    @staticmethod
    def audio_process(audio_raw):
        audio_resample = librosa.core.resample(audio_raw, orig_sr=source_frequency, target_sr=target_frequency)
        audio_resample *= 1. / max_audio * 0.9
        audio_fft = librosa.stft(y=audio_resample, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        audio_fft = np.abs(audio_fft)
        audio_db = librosa.core.amplitude_to_db(audio_fft)
        audio_db -= audio_mean
        audio_db /= audio_std
        return audio_db

    def velocity_change(self):
        pass


def main():
    rospy.init_node("robot_pouring_demo")
    while not rospy.is_shutdown():
        Pouring(target_cavity_height=args.cavity_height)


if __name__ == "__main__":
    main()
