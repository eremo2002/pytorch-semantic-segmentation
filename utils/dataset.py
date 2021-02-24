import numpy as np
import os
import torch
import cv2
import torchvision.transforms.functional as F
from PIL import Image

class VOC_segmentation(torch.utils.data.Dataset):
    def __init__(self, image_set, transform):        
        super(VOC_segmentation, self).__init__()

        self.image_set = image_set        
        self.transform = transform
        
        
        image_dir = os.path.join('../data/VOCdevkit/VOC2012/JPEGImages')
        label_dir = os.path.join('../data/VOCdevkit/VOC2012/SegmentationClass')
        
        splits_dir = os.path.join( '../data/VOCdevkit/VOC2012/ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.labels = [os.path.join(label_dir, x + ".png") for x in file_names]

        assert (len(self.images) == len(self.labels))
    

    def __getitem__(self, index):
        # output would have the shape [batch_size, nb_classes, height, width]
        # The target would have the shape [batch_size, height, width] and contain values in the range [0, nb_classes-1]
        # https://discuss.pytorch.org/t/tusimple-enet-lane-binary-segmentation-to-multi-channel-output-segmentation/82800/3

        image = Image.open(self.images[index])
        label = Image.open(self.labels[index])
        
        image = np.array(image, dtype=np.float32)
        image = image[:,:,::-1] # bgr to rgb
        label = np.array(label, dtype=np.uint8)
        label[label==255] = 0

        data = {'image': image, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.images)


class ToTensor(object):  
    def __call__(self, data):        
        image, label = data['image'], data['label']
        image = image.transpose((2, 0, 1)).astype(np.float32)        
        image = torch.from_numpy(image)        

        label = label.astype(np.uint8) # if no this line, error
        label = torch.from_numpy(label)       
        label = label.clone().detach().long()
        
        data = {'image': image, 'label': label}

        return data


class Resize(object):
    def __init__(self, size=(224, 224)):
        self.size = size
    
    def __call__(self, data):
        image, label = data['image'], data['label']
        image = cv2.resize(image, self.size)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)        

        data = {'image': image, 'label': label}
        return data
    


# image normalize [0, 1] and not apply to label
class Normalize(object):   
    def __call__(self, data):
        image, label = data['image'], data['label']
        # print('before normalize', np.min(image), np.max(image))
        image = image/255.0
        # print('after normalize', np.min(image), np.max(image))
        
        data = {'image': image, 'label': label}

        return data


class Horizontal_flip(object):
    def __call__(self, data):
        image, label = data['image'], data['label']

        if np.random.rand() >= 0.5:
            image = image[:, ::-1, :]
            label = label[:, ::-1]

        data = {'image': image, 'label': label}
        return data


class Vertical_flip(object):   
    def __call__(self, data):
        image, label = data['image'], data['label']

        if np.random.rand() >= 0.5:
            image = image[::-1, :, :]
            label = label[::-1, :]
        
        data = {'image': image, 'label': label}

        return data

class Cutout(object):
    def __init__(self, mask_ratio=0.3):
        self.mask_ratio = mask_ratio
        self.mask_value = 0
    
    def __call__(self, data):
        image, label = data['image'], data['label']        

        top = np.random.randint(0, image.shape[0]*self.mask_ratio)
        left = np.random.randint(0, image.shape[1]*self.mask_ratio)
        bottom = int(top + (image.shape[0]*self.mask_ratio))
        right = int(left + (image.shape[1]*self.mask_ratio))
        
        

        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > image.shape[0]:
            bottom = image.shape[0]
        if right > image.shape[1]:
            right = image.shape[1]
            
        image[top:bottom, left:right, :].fill(self.mask_value)
        label[top:bottom, left:right].fill(self.mask_value)

        data = {'image': image, 'label': label}

        return data
