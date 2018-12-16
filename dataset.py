from PIL import Image
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.models as models
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
        img_file = filename
        img = Image.open(img_file)
        img=img.resize((224,224))
        img = self.transform(img)
        return img, label


if __name__ == '__main__':
    My_PATH = '/media/x/287464E27464B46A/linuxhome/datasets/all'
    ds_train = MyDataset(My_PATH, set='train')
    loader_train = data_utils.DataLoader(ds_train,
                                         batch_size=1,
                                         num_workers=1, shuffle=True)
    model = models.vgg16(pretrained=True)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 2))

    print("init_params done.")
    model = model.cuda(0)
    print('finish')

    for i, (images, label) in enumerate(loader_train):
        print(i)
### EOF ###
