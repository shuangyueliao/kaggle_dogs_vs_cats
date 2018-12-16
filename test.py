import argparse
import csv
import torchvision.transforms as transforms
from PIL import Image
import torch, datetime
import os
from torch.autograd import Variable
import torchvision.models as models

prePath = '/media/x/287464E27464B46A/linuxhome/datasets/all/test/'
#prePath = 'C:/Users/Administrator/Desktop/data/test/'

def train(args):
    print(args)
    model=torch.load('./models/2.pkl')
    model=model.cuda(0)
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    with open('./sample_submission.csv', 'w', newline='') as myFile:
        myWriter = csv.writer(myFile)
        tmp = []
        tmp.append('id')
        tmp.append('label')
        myWriter.writerow(tmp)
        for root, dirs, files in os.walk(prePath):
            for file in files:
                img = Image.open(prePath+file)
                img = img.resize((224, 224))
                images = transform(img)
                images=torch.unsqueeze(images,0)
                images=images.cuda(0)
                images = Variable(images)
                outputs = model(images)
                _, pred = torch.max(outputs.data, 1)
                tmp = []
                tmp.append(file.split('.')[0])
                tmp.append(int(pred))
                myWriter.writerow(tmp)
                print(file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_worker', type=int, default=4,
                        help='# of workers')
    args = parser.parse_args()
    train(args)
