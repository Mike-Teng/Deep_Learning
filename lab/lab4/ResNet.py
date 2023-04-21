import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from dataloader import RetinopathyLoader

class ResNet18(nn.Module):
    def __init__(self, num_class, use_pretrained):
        super(ResNet18,self).__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT' if use_pretrained else None)
        fc_num_neurons = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_num_neurons, num_class)

    def forward(self, x):
        x = self.model(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_class, use_pretrained):
        super(ResNet50,self).__init__()
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT' if use_pretrained else None)
        fc_num_neurons = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_num_neurons, num_class)

    def forward(self, x):
        x = self.model(x)
        return x

def evaluate(model, test_loader, device, num_class):
    confusion_matrix=np.zeros((num_class,num_class))
    with torch.set_grad_enabled(False):
        model.eval()
        correct = 0
        label_list = []
        pred_list = []
        for _, (images, label) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            label_list.extend(label)

            predict = model(images)
            pred = predict.argmax(dim=1)
            pred_list.extend(pred)
            
            for i in range(len(label)):
                # confusion_matrix[int(label[i])][int(pred[i])]+=1
                if pred[i] == label[i]:
                    correct +=1
        cf_matrix = confusion_matrix(label_list, pred_list)
        acc = 100. * correct / len(test_loader.dataset)
    #normalize
    # confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(num_class,1)
    return acc, cf_matrix


epochs = 20   
momentum = 0.9
weight_decay = 5e-4
lr = 0.0008

def train(model, train_loader, test_loader, optimizer, device, num_class, name_type):
    df = pd.DataFrame(columns=['acc_train','acc_test'])
    best_model_weights=None
    best_evaluated_acc=0
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(1, epochs+1):
        with torch.set_grad_enabled(True):
            model.train()
            total_loss = 0
            total_correct = 0
            for index, (images, label) in enumerate(train_loader):
                images = images.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)
                predict = model(images)
                loss = criterion(predict, label)
                total_loss += loss.item()
                pred = predict.argmax(dim=1)

                # total_correct = sum(x == y for x, y in zip(label, pred))
                for i in range(len(label)):
                    if pred[i] == label[i]:
                        total_correct +=1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss = total_loss / len(train_loader.dataset)
            acc_train = total_correct / len(train_loader.dataset)

            df.loc[epoch, 'acc_train'] = acc_train
            print(f'epoch{epoch:>2d} loss:{total_loss:.4f} acc_train:{acc_train:.3f}')
        
        acc_test, _ = evaluate(model, test_loader, device, num_class)
        df.loc[epoch, 'acc_test'] = acc_test

        print(f'epoch{epoch:>2d}  acc_test:{acc_test:.3f}%')
        if acc_test > best_evaluated_acc:
            best_evaluated_acc = acc_test
            torch.save(model.state_dict(), os.path.join('./models', name_type+'.pt'))

    print("best acc ", name_type, ": ", best_evaluated_acc)

    # save model
    model.load_state_dict(best_model_weights)
    return df

if __name__== "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.__version__, ", using", device)
    
    batch_size_18 = 16
    batch_size_50 = 8
    num_class= 5

    train_dataset = RetinopathyLoader('./new_train', mode="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size_18, shuffle=True, num_workers=4)
    test_dataset = RetinopathyLoader('./new_test', mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size_18, shuffle=True, num_workers=4)

    # print('resnet50 with pretrained')
    # model_with_50 = ResNet50(num_class, use_pretrained=True)    
    # optimizer=optim.SGD(model_with_50.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # df_with_pretrain_50 = train(model_with_50, train_loader, test_loader, optimizer=optimizer, device=device, num_class=num_class, name_type='resnet50_with_pretrain')
    # df_with_pretrain_50.to_csv('./models/resnet50_with_pretrain.csv', index=False)


    print('resnet18 with pretrained')
    model_with_18 = ResNet18(num_class, use_pretrained=True)    
    optimizer=optim.SGD(model_with_18.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    df_with_pretrain_18 = train(model_with_18, train_loader, test_loader, optimizer=optimizer, device=device, num_class=num_class, name_type='resnet18_with_pretrain')
    df_with_pretrain_18.to_csv('./models/resnet18_with_pretrain.csv', index=False)