import os,sys
import argparse

import numpy as np
import logging
import torch, datetime
import torch.nn.functional as F
import torch.utils.data as data_utils

from torch.autograd import Variable

import torchvision.models as models

from dataset import MyDataset
My_PATH='/media/x/287464E27464B46A/linuxhome/datasets/all'
def train(args):
    print(args)
    ds_train = MyDataset(My_PATH, set='train')
    ds_val   = MyDataset(My_PATH, set='val')
    loader_train = data_utils.DataLoader(ds_train,
                                         batch_size=args.batch_size,
                                         num_workers=args.nb_worker,shuffle=True)
    loader_val = data_utils.DataLoader(ds_val,
                                       batch_size=1,
                                       num_workers=1,shuffle=False)

    model=torch.load('./models/2.pkl')
    model=model.cuda(0)

    trainAcc = 0
    trainNum = ds_train.__len__()
    for i, (images, label) in enumerate(loader_train):
        images = images.cuda(0)
        label = label.cuda(0)

        images = Variable(images)
        label = Variable(label)

        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        trainAcc += torch.sum(pred == label.data)
        print(i)
        print(trainAcc)
    print('-----------------------------')
    valAcc = 0
    for i, (images, label) in enumerate(loader_val):
        images=images.cuda(0)
        images = Variable(images)
        label=label.cuda(0)

        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        valAcc += torch.sum(pred == label)
        print(i)
        print(valAcc)
    print("Epoch [%d/%d],trainAcc: %.4f,valAcc: %.4f" % (1, args.nb_epoch,  int(trainAcc)* 1.0 / trainNum, int(valAcc)* 1.0 / (i + 1)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epoch', type=int, default=150,
                        help='# of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--nb_worker', type=int, default=4,
                        help='# of workers')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning Rate')
    args = parser.parse_args()
    logging.basicConfig(filename='about.log', level=logging.INFO, filemode='a+')
    train(args)


### EOF ###
