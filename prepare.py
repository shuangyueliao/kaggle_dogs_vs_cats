# -*- coding: utf-8 -*-
import os

file_dir = 'C:/Users/Administrator/Desktop/data/train'
prePath = '/home/xxx/'
train_files = []
val_files = []
test_files = []
cat_files = []
dog_files = []
for root, dirs, files in os.walk(file_dir):
    for file in files:
        if file.find('cat') != -1:
            cat_files.append(file)
        else:
            dog_files.append(file)
print(cat_files)
print(dog_files)
with open('train', 'w') as f1, open('val', 'w') as f2:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    traincatlen = int(len(cat_files) * 0.8)
    for index, i in enumerate(cat_files):
        if index <= traincatlen:
            f1.write(prePath + i + '@0\n')  # 猫是0，狗是1
        else:
            f2.write(prePath + i + '@0\n')  # 猫是0，狗是1
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
