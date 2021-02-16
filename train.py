import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import math
from models.FCN import FCN8, FCN16
from models.DeconvNet import DeconvNet
from models.SegNet import SegNet
from models.UNet import UNet
from utils.dataset import *

os.environ['CUDA_VISIBLE_DEVICES'] ='0'
print(torch.cuda.is_available())
device = torch.device('cuda:0')
print(device)


# hyper parameters
batch_size = 16
epochs = 300


transform_train = transforms.Compose([		
		Resize((224, 224)),
		Normalize(),
		ToTensor()		
		])

transform_val = transforms.Compose([		
		Resize((224, 224)),
		Normalize(),
		ToTensor()
		])

dataset_train = VOC_segmentation('train', transform = transform_train)
dataloader_train = torch.utils.data.DataLoader(dataset_train, 
										batch_size=batch_size, 
										shuffle=True,
										# num_workers=1
										)

dataset_val = VOC_segmentation('val', transform = transform_val)
dataloader_valid = torch.utils.data.DataLoader(dataset_val, 
										batch_size=4, 
										shuffle=False,
										# num_workers=1
										)




# model
# model = FCN16(num_classes=21)
model = DeconvNet(num_classes=21)
x = torch.randn([1, 3, 224, 224])
out = model(x)
# print('input shape : ', x.shape)
# print('output shape : ', out.size())
model.to(device)


# loss function
criterion = nn.CrossEntropyLoss().to(device)


# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)


# 
print(f'train images {len(dataset_train)}, val images {len(dataset_val)}')



best_val_loss = 999999
for epoch in range(epochs):	
	train_loss = 0.0
	for step, data in enumerate(dataloader_train):		
		image = data['image'].to(device)
		label = data['label'].to(device)		
		# print(image.shape, label.shape)
		outputs = model(image)				
		# outputs [N, 21, 224, 224]
		# label [N, 224, 224]

		# backwards
		loss = criterion(outputs, label)			
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		if (step + 1) % 10 == 0:
			print(f'Epoch: {epoch+1}, Step: {step+1}, Loss: {train_loss/10:.8f}')
			train_loss = 0.0

	# validation
	model.eval()
	with torch.no_grad():
		val_loss_list = []		
		for step, data in enumerate(dataloader_valid):
			image = data['image'].to(device)
			label = data['label'].to(device) 			

			outputs = model(image)
			loss = criterion(outputs, label)
			val_loss_list.append(loss.item())

		val_loss = sum(val_loss_list)/len(val_loss_list)
		print(f'Epoch: {epoch+1}, val_loss: {val_loss:.4f}')
	
		if best_val_loss > val_loss:
			print(f'Epoch: {epoch+1}, val_loss is improved from {best_val_loss:.4f} to {val_loss:.4f}')
			best_val_loss = val_loss

			# save best model
			torch.save({
			'epoch': epochs,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': criterion,
			}, 
			'./weights/model.pth')
			

print('Finished Training')

