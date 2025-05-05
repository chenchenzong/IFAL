import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
import os
from PIL import Image

        
# For Tiny-Imagenet
class TinyImageNet(Dataset):
    def __init__(self, type, transform, labels, image_names):
        self.type = type
        if type == 'train':
            i = 0
            self.data = []
            for label in labels:
                image = []
                for image_name in image_names[i]:
                    image_path = os.path.join('../tiny-imagenet-200/train', label, 'images', image_name) 
                    data = Image.open(image_path).convert("RGB")
                    image.append(np.asarray(data))
                    data.close()
                self.data.append(image)
                i = i + 1
            self.data = np.array(self.data)
            self.data = self.data.reshape(-1, 64, 64, 3)
            self.uq_idxs = range(self.data.shape[0])
            self.targets = [i//500 for i in self.uq_idxs]
            self.uq_idxs = np.asarray(self.uq_idxs)
        elif type == 'val':
            self.data = []
            for image in image_names:
                image_path = os.path.join('../tiny-imagenet-200/val/images', image)
                data = Image.open(image_path).convert("RGB")
                self.data.append(np.asarray(data))
                data.close()
            self.data = np.array(self.data)
            self.uq_idxs = range(self.data.shape[0])
            self.targets = labels
            self.uq_idxs = np.asarray(self.uq_idxs)
        self.ToPILImage = transforms.ToPILImage()
        self.transform = transform
        
    def __getitem__(self, index):
        label = []
        image = []
        if self.type == 'train':
            label = self.targets[index]
            image = self.data[index]
            image = self.ToPILImage(np.uint8(image))
        if self.type == 'val':
            label = self.targets[index]
            image = self.data[index]
            image = self.ToPILImage(np.uint8(image))
        return self.transform(image), label
        
    def __len__(self):
        len = 0
        if self.type == 'train':
            len = self.data.shape[0]
        if self.type == 'val':
            len = self.data.shape[0]
        return len  


    
def get_tinyimagenet():
    print(os.getcwd())
    num_classes = 200
        
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        

    labels_t = []
    image_names = []
    with open('../tiny-imagenet-200/wnids.txt') as wnid:
        for line in wnid:
            labels_t.append(line.strip('\n'))
    for label in labels_t:
        txt_path = '../tiny-imagenet-200/train/'+label+'/'+label+'_boxes.txt'
        image_name = []
        with open(txt_path) as txt:
            for line in txt:
                image_name.append(line.strip('\n').split('\t')[0])
        image_names.append(image_name)
    labels = np.arange(200)

    val_labels_t = []
    val_labels = []
    val_names = []
    with open('../tiny-imagenet-200/val/val_annotations.txt') as txt:
        for line in txt:
            val_names.append(line.strip('\n').split('\t')[0])
            val_labels_t.append(line.strip('\n').split('\t')[1])
    for i in range(len(val_labels_t)):
        for i_t in range(len(labels_t)):
            if val_labels_t[i] == labels_t[i_t]:
                val_labels.append(i_t)
    val_labels = np.array(val_labels)
        
    dataset_train = TinyImageNet("train", train_transform,labels_t,image_names)
    dataset_query = TinyImageNet("train", test_transform,labels_t,image_names)
    dataset_test = TinyImageNet("val", test_transform,val_labels,val_names)
        
    return dataset_train, dataset_query, dataset_test

