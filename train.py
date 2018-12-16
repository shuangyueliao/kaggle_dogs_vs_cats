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
    model = models.vgg16(pretrained = True)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 2))

    print ("init_params done.")
    model=model.cuda(0)
    if not os.path.exists("./models"):
        os.mkdir ("./models")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cost=torch.nn.CrossEntropyLoss()
    cost=cost.cuda(0)
    for epoch in range(args.nb_epoch):
        starttime = datetime.datetime.now()
        trainAcc = 0
        trainNum = ds_train.__len__()
        for i, (images, label) in enumerate(loader_train):
            images = images.cuda(0)
            label = label.cuda(0)

            images = Variable(images)
            label = Variable(label)

            optimizer.zero_grad()
            outputs = model(images)
            loss = cost(outputs, label)
            _, pred = torch.max(outputs.data, 1)
            trainAcc += torch.sum(pred == label.data)
            loss.backward()
            optimizer.step()

        valAcc = 0
        for i, (images, label) in enumerate(loader_val):
            images=images.cuda(0)
            images = Variable(images)
            label=label.cuda(0)

            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            valAcc += torch.sum(pred == label)

        print("Epoch [%d/%d] Loss: %.6f,trainAcc: %.4f,valAcc: %.4f" % (
        epoch + 1, args.nb_epoch, loss.data[0], int(trainAcc)* 1.0 / trainNum,int(valAcc)* 1.0 / (i + 1)))
        logging.info("Epoch [%d/%d] Loss: %.6f,trainAcc: %.4f,valAcc: %.4f" % (
        epoch + 1, args.nb_epoch, loss.data[0],int(trainAcc)* 1.0 / trainNum,int(valAcc)* 1.0 / (i + 1)))
        torch.save(model, "./models/{}.pkl".format(epoch))
        endtime=datetime.datetime.now()
        print((endtime-starttime).seconds)
    logging.info('time:{}'.format((endtime-starttime).seconds))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epoch', type=int, default=5,
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
