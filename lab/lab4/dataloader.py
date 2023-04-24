import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        single_img_name = os.path.join(self.root, self.img_name[index]+ '.jpeg')
        single_img =  Image.open(single_img_name)

        # images resize
        width, height = single_img.size 
        single_img = transforms.CenterCrop(height)(single_img)
        single_img = transforms.Resize((512,512))(single_img)

        # images process
        single_img = transforms.RandomHorizontalFlip(p=0)(single_img)
        single_img = transforms.RandomVerticalFlip(p=0)(single_img)
        single_img = transforms.ToTensor()(single_img)

        # images normalization
        # mean_list = np.array([torch.mean(single_img[0]), torch.mean(single_img[1]), torch.mean(single_img[2])])
        # std_list = np.array([torch.std(single_img[0]), torch.std(single_img[1]), torch.std(single_img[2])])
        mean_list = [0.443162, 0.306861, 0.21887]
        std_list =  [0.202909, 0.141089, 0.09971]
        img = transforms.Normalize(mean_list, std_list)(single_img)

        label = self.label[index]

        return img, label