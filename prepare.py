# -*- coding: utf-8 -*-
import os

prePath = '/media/x/287464E27464B46A/linuxhome/datasets/all/train/'
prePathTxt = '/media/x/287464E27464B46A/linuxhome/datasets/all/'
train_files = []
val_files = []
test_files = []
cat_files = []
dog_files = []
for root, dirs, files in os.walk(prePath):
    for file in files:
        if file.find('cat') != -1:
            cat_files.append(file)
        else:
            dog_files.append(file)
print(cat_files)
print(dog_files)
with open(prePathTxt+'train.txt', 'w') as f1, open(prePathTxt+'val.txt', 'w') as f2:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    traincatlen = int(len(cat_files) * 0.8)
    for index, i in enumerate(cat_files):
        if index <= traincatlen:
            f1.write(prePath + i + '@0\n')  # 猫是0，狗是1
        else:
            f2.write(prePath + i + '@0\n')  # 猫是0，狗是1
        print('cat:',i)
    traindoglen = int(len(dog_files) * 0.8)
    for index, i in enumerate(dog_files):
        if index <= traindoglen:
            if index == traindoglen:
                f1.write(prePath + i + '@1')  # 猫是0，狗是1
            else:
                f1.write(prePath + i + '@1\n')  # 猫是0，狗是1
        else:
            if index == len(dog_files) - 1:
                f2.write(prePath + i + '@1')  # 猫是0，狗是1
            else:
                f2.write(prePath + i + '@1\n')  # 猫是0，狗是1
        print('dog:',i)
