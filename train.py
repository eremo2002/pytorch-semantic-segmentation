import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import math
from models.FCN import FCN16
from models.DeconvNet import DeconvNet
from utils.dataset import VOC_segmentation

os.environ['CUDA_VISIBLE_DEVICES'] ='0'
print(torch.cuda.is_available())
device = torch.device('cuda:0')
print(device)


# custom dataset
transform_train = transforms.Compose([
        transforms.Resize((224, 224)),        
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])        
        ])
transform_target = transforms.Compose([
        transforms.Resize((224, 224)),        
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

dataset = VOC_segmentation('train', 
                            train_transform = transform_train, 
                            target_transform=transform_target)

dataloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=16, 
                                        shuffle=True,
                                        num_workers=0)


classes = ('background',
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor')

print(len(dataset))


# model = FCN16(21)
model = DeconvNet(21)


# # multi-gpu
# net = nn.DataParallel(net)

model.to(device)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(100):   #
    train_loss = 0.0
    for i, data in enumerate(dataloader):        
        inputs, labels = data[0].to(device), data[1].to(device)
        #print(inputs.shape, labels.shape)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if i % 16 == 15:    
            print('[%d,\t%5d] loss: %.4f' %
                  (epoch + 1, i + 1, train_loss / 16))
            train_loss = 0.0

print('Finished Training')
