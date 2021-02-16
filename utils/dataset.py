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

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
                (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask   

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    

    def __getitem__(self, index):
        # target would have the shape [batch_size, nb_classes, height, width]
        # https://discuss.pytorch.org/t/tusimple-enet-lane-binary-segmentation-to-multi-channel-output-segmentation/82800/3

        image = Image.open(self.images[index])
        label = Image.open(self.labels[index])
        
        image = np.array(image, dtype=np.float32)        
        label = np.array(label, dtype=np.uint8)
        label[label==255] = 0        
        label = self.encode_segmap(label)    
        
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
        image = image/255.0
        image = torch.from_numpy(image)

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
