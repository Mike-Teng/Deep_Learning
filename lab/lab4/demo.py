import torch
from dataloader import RetinopathyLoader
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ResNet import ResNet18, ResNet50, plot_cf_matrix, evaluate



batch_size_18 = 16
batch_size_50 = 8

if __name__== "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = RetinopathyLoader(root='./new_test', mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size_50, shuffle=False, num_workers=4)

    model = ResNet50(num_class=5, use_pretrained=True)
    model.load_state_dict(torch.load(os.path.join('./models', '828_resnet50_with_pretrain.pt')))
    model = model.to(device)
    acc,_ = evaluate(model, test_loader, device, num_class=5)
    print(f"resnet50_with_pretrain: {acc*100:.2f} %")
    
    
    # model = ResNet18(num_class=5, use_pretrained=False)
    # model.load_state_dict(torch.load(os.path.join('./models', 'resnet18_wo_pretrain.pt')))
    # model = model.to(device)
    # acc,_ = evaluate(model, test_loader, device, num_class=5)
    # print(f"resnet18_without_pretrain: {acc*100:.2f} %")