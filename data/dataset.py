import os
import csv
from PIL import Image
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, opt, phase, transform=None):
        self.data_dir = opt.data_dir
        self.data_name = opt.data_name
        self.transform = transform

        self.data = list()
        with open(os.path.join(self.data_dir, self.data_name, '{}.csv'.format(phase))) as f:
            reader = csv.reader(f)
            for line in reader:
                self.data.append(line)

        self.label2num = {}
        with open(os.path.join(self.data_dir, self.data_name, 'label.txt'), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.label2num[line.strip()] = i

        self.img_names = list(map(lambda x: x[0], self.data))
        self.labels = list(map(lambda x: x[1], self.data))
        self.labels = list(map(lambda x: self.label2num[x], self.labels))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.data_name, self.img_names[index]))
        img = img.convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.data)