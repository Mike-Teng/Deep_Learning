from torchvision import transforms
import numpy as np
from PIL import Image
from dataloader import RetinopathyLoader
from torch.utils.data import Dataset, DataLoader


train_dataset = RetinopathyLoader('./new_train', mode="train")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
mean_total =  np.zeros(3)
std_total = np.zeros(3)
for i, (images, label, mean_list, std_list) in enumerate(train_loader):
    print(i)
    mean_list = mean_list.numpy()[0]
    std_list = std_list.numpy()[0]

    mean_total += mean_list
    std_total += std_list
    mean_total = [mean_total[0]+mean_list[0], mean_total[1]+mean_list[1], mean_total[2]+mean_list[2]]
    std_total = [std_total[0]+std_list[0], std_total[1]+std_list[1], std_total[2]+std_list[2]]

mean_total = mean_total / len(train_loader.dataset)
std_total = std_total / len(train_loader.dataset)
print('mean_total', mean_total)
print('std_total', std_total)

