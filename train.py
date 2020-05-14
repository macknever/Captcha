########################################
### 2020/05/10 Lawrence
### Train for captcha
########################################


import torch
import torch.nn as nn
from torch.autograd import Variable
from model import captcha_identifier
from dataloader_captcha import Captcha_Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms,models

import math
import time
import os

from mean_std_obtainer import get_mean_std

base_lr = 0.001
max_epoch = 100
model_path = './checkpoints/identifier.pth'
restor = False

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

def calculat_acc(output, target):
    output, target = output.view(-1, 36), target.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc

def train():
        ## 
    mean,std = get_mean_std('captcha')

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean,std = std)])
    ## path = "data/"
    dataset = Captcha_Dataset('captcha', transform = transform)
    
    trainidx = 0
    validx = int(math.floor(len(dataset) * 0.9))

    train_set = torch.utils.data.Subset(dataset, list(range(0, validx)))
    val_set = torch.utils.data.Subset(dataset, list(range(validx, len(dataset))))
    
    train_loader = torch.utils.data.DataLoader(train_set, 
        batch_size = 128, 
        num_workers = 0,
        pin_memory = False,
        shuffle = True,
        drop_last = False)
    validation_loader = torch.utils.data.DataLoader(val_set, 
        batch_size = 6, 
        num_workers = 0,
        pin_memory = False,
        shuffle = True,
        drop_last = False)


    identifier = captcha_identifier()
    #identifier.load_state_dict(torch.load(model_path))


    model = models.resnet18(pretrained = False)
#     if torch.cuda.is_available():
#         cnn.cuda()
#     if restor:
#         cnn.load_state_dict(torch.load(model_path))
# #        freezing_layers = list(cnn.named_parameters())[:10]
# #        for param in freezing_layers:
# #            param[1].requires_grad = False
# #            print('freezing layer:', param[0])
    
    optimizer = torch.optim.Adam(identifier.parameters(), lr=base_lr)
    loss_fn = nn.MultiLabelSoftMarginLoss()
    
    losses = []

    for epoch in range(max_epoch):
        
        train_iter = iter(train_loader)
        for i in range(len(train_loader)):
            batch = next(train_iter)
            batch = Variable(batch)
            preds = identifier(batch)
            
            loss = loss_fn(preds,batch['target'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            torch.save(identifier.state_dict(),model_path)
        
        
            
            if i % 10 == 0:
                print("Loss:", i, losses[-1])

            
            
        

if __name__=="__main__":
    train()
    pass




