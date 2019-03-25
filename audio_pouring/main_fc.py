#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name     : main.py
# Purpose       : fc
# Creation Date : 05-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]

import argparse
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as torch_func
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from model.dataset import PouringDataset
from model.model import AudioCNNRealT

parser = argparse.ArgumentParser(description='audio2height')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fixed', action='store_true')
parser.add_argument('--bottle', type=str, default='1')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--load-model', type=str, default='')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                    help='pre-trained model path')
parser.add_argument('--data-path', type=str, default="dataset/", help='data path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=5)
args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available() else False

if args.cuda:
    torch.cuda.manual_seed(1)

logger = SummaryWriter(os.path.join('./assets/log/', args.tag))
np.random.seed(int(time.time()))


def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2 ** 31 - 1))


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


input_audio_size = 257
thresh_acc = [1, 2, 3, 4, 5]  # unit mm
bottle_upper = torch.tensor([150.0])
bottle_lower = torch.tensor([0.0])
multi_modal = False

train_loader = torch.utils.data.DataLoader(
    PouringDataset(
        path=args.data_path,
        input_audio_size=input_audio_size,
        multi_modal=multi_modal,
        train_rnn=False,
        is_fixed=args.fixed,
        is_train=True,
        bottle_train=args.bottle,
        bottle_test=args.bottle
    ),
    batch_size=args.bs,
    drop_last=True,
    num_workers=128,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    # collate_fn=my_collate,
)

test_loader = torch.utils.data.DataLoader(
    PouringDataset(
        path=args.data_path,
        input_audio_size=input_audio_size,
        multi_modal=multi_modal,
        train_rnn=False,
        is_fixed=args.fixed,
        is_train=False,
        bottle_train=args.bottle,
        bottle_test=args.bottle
    ),
    batch_size=args.bs,
    num_workers=128,
    drop_last=True,
    pin_memory=True,
    shuffle=False,
    worker_init_fn=worker_init_fn,
    # collate_fn=my_collate,
)

is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1

if is_resume or args.mode == 'test':
    model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
    model.device_ids = [args.gpu]
    print('load model {}'.format(args.load_model))
else:
    model = AudioCNNRealT(feature_size=input_audio_size, hidden_size=128)

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [1, 0]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    bottle_upper = bottle_upper.cuda()
    bottle_lower = bottle_lower.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=80, gamma=0.5)


def train(model_, loader, epoch):
    scheduler.step()
    model_.train()
    torch.set_grad_enabled(True)
    train_error = 0
    correct_height = [0] * len(thresh_acc)

    data_num = 0
    for batch_idx, (audio, target) in enumerate(loader):
        if args.cuda:
            audio, target = audio.cuda(), target.cuda()
        optimizer.zero_grad()

        # audio = audio.squeeze()[0:-1,:].reshape(-1,args.audio_bs,input_audio_size)
        height = model_(audio)
        height = height * (bottle_upper - bottle_lower) + bottle_lower
        target = target.view(-1, 1)
        loss = torch_func.mse_loss(height, target)
        loss.backward()
        optimizer.step()

        # compute acc
        is_correct = [abs(height.cpu().data.numpy() - target.cpu().data.numpy()) < thresh
                      for thresh in thresh_acc]
        res_acc = [np.sum(cc) for cc in is_correct]
        correct_height = [c + r for c, r in zip(correct_height, res_acc)]
        data_num += target.shape[0]
        # compute average error
        train_error += torch_func.l1_loss(height, target, size_average=False)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttag: {}'.
                  format(epoch, batch_idx * args.bs, len(loader.dataset),
                         100. * batch_idx * args.bs / len(loader.dataset),
                         loss.item(), args.tag))
            logger.add_scalar('train_loss', loss.item(), batch_idx + epoch * len(loader))

    train_error /= float(data_num)
    acc_height = [float(c) / float(data_num) for c in correct_height]
    return acc_height, train_error


def test(model_, loader):
    model_.eval()
    torch.set_grad_enabled(False)
    test_error = 0
    correct_height = [0] * len(thresh_acc)
    data_num = 0
    test_loss = 0
    res = []
    for batch_idx, (audio, target) in enumerate(loader):
        if args.cuda:
            audio, target = audio.cuda(), target.cuda()

        # audio = audio.squeeze()[0:-1,:].reshape(-1,args.audio_bs,input_audio_size)
        height = model_(audio)
        height = height * (bottle_upper - bottle_lower) + bottle_lower

        target = target.view(-1, 1)
        test_loss += torch_func.mse_loss(height, target)

        # compute acc
        is_correct = [abs(height.cpu().data.numpy() - target.cpu().data.numpy()) < thresh
                      for thresh in thresh_acc]
        res_acc = [np.sum(cc) for cc in is_correct]
        correct_height = [c + r for c, r in zip(correct_height, res_acc)]
        data_num += target.shape[0]
        # compute average error
        test_error += torch_func.l1_loss(height, target, size_average=False)
        res.append(target)

    test_error /= float(data_num)
    test_loss = test_loss / len(loader)
    acc_height = [float(c) / float(data_num) for c in correct_height]
    f = open('b3_label.csv', 'w')
    for batch in res:
        buf = [str(i[0]) for i in batch.cpu().data.numpy()]
        f.write(','.join(buf) + '\n')
    return acc_height, test_error, test_loss


def main():
    if args.mode == 'train':
        for epoch in range(is_resume * args.load_epoch, args.epoch):
            acc_train, train_error = train(model, train_loader, epoch)
            print('Train done, acc={}, train_error={}'.format(acc_train, train_error))
            acc_test, test_error, test_loss = test(model, test_loader)
            print('Test done, acc_test={}, test_error ={}, test_loss={}'.
                  format(acc_test, test_error, test_loss))

            logger.add_scalar('train_acc1', acc_train[0], epoch)
            logger.add_scalar('train_acc2', acc_train[1], epoch)
            logger.add_scalar('train_acc3', acc_train[2], epoch)
            logger.add_scalar('train_acc4', acc_train[3], epoch)
            logger.add_scalar('train_acc5', acc_train[4], epoch)

            logger.add_scalar('test_acc1', acc_test[0], epoch)
            logger.add_scalar('test_acc2', acc_test[1], epoch)
            logger.add_scalar('test_acc3', acc_test[2], epoch)
            logger.add_scalar('test_acc4', acc_test[3], epoch)
            logger.add_scalar('test_acc5', acc_test[4], epoch)

            logger.add_scalar('test_error', test_error, epoch)
            logger.add_scalar('train_error', train_error, epoch)

            logger.add_scalar('test_loss', test_loss, epoch)

            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc_test, test_error, test_loss = test(model, test_loader)
        print('Test done, acc_test={}, test_error ={}, test_loss={}'.
              format(acc_test, test_error, test_loss))


if __name__ == "__main__":
    main()
