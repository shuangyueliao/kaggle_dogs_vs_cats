#from torch.utils import data

import os,sys
import numpy as np

import scipy.misc as m
from PIL import Image
import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class MyDataset(data_utils.Dataset):
    def __init__(self, root_path, set='train', img_size=224):
        self.root_path = root_path
        self.set       = set
        self.img_size  = img_size
        transform_list = [
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        self.files = []
        with open (self.root_path + '/' + self.set + '.txt', 'r') as f:
            for line in f:
                self.files.append(line.rstrip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        base_name = self.files[index]
        tmp=base_name.split('@')
        #print(tmp)
        label=tmp[1]
        filename=tmp[0]
        label=int(label[0])
        img_file = self.root_path + '/' +self.set+'/'+ filename
        img = Image.open(img_file)
        img=img.resize((224,224))
        img = self.transform(img)
        return img, label


if __name__ == '__main__':
    My_PATH = 'C:/Users/Administrator/Desktop/data'
    ds_train = MyDataset(My_PATH, set='train')
    loader_train = data_utils.DataLoader(ds_train,
                                         batch_size=16,
                                         num_workers=4, shuffle=True)
    print(ds_train.__len__())
    for image, label in loader_train:
        print(type(image))
        break

### EOF ###
