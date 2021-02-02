import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class FCN16(nn.Module):
    def __init__(self, num_classes):
        super(FCN16, self).__init__()

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=False)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=False)

        # pool1
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=False)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=False)        
        
        # pool2
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=False)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=False)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=False)
        
        # pool3
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=False)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=False)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=False)
        
        # pool4
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=False)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=False)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=False)
        
        # pool5
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=False)
        self.drop6 = nn.Dropout(0.5, inplace=False)

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=False)
        self.drop7 = nn.Dropout(0.5, inplace=False)

        # upsampling for final output. 'score' means channel resizing for K class 
        self.score_pool4 = nn.Conv2d(512, num_classes, 1) # pool4 channel resize
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=16, bias=False)         
        

    def forward(self, x):
        input_h, input_w = x.size()[2], x.size()[3]

        x = self.conv1_1(x)
        x = self.relu1_1(x)  
        x = self.conv1_2(x)
        x = self.relu1_2(x)  
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)  
        x = self.conv2_2(x)
        x = self.relu2_2(x)  
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)  
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)  
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.relu4_1(x)  
        x = self.conv4_2(x)
        x = self.relu4_2(x)  
        x = self.conv4_3(x)
        x = self.relu4_3(x)  
        pool4 = self.pool4(x)

        x = self.conv5_1(pool4)
        x = self.relu5_1(x)  
        x = self.conv5_2(x)
        x = self.relu5_2(x)  
        x = self.conv5_3(x)
        x = self.relu5_3(x) 
        x = self.pool5(x)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        
        x = self.score_fr(x)
        upscore2 = self.upscore2(x)

        pool4 = self.score_pool4(pool4)       


        # print(f'upscore2 {upscore2.shape},   pool4 {pool4.shape}')
        dh, dw = (pool4.size()[2] - upscore2.size()[2])//2, (pool4.size()[3] - upscore2.size()[3])//2        
        # print(dh, dw)        
        upscore16 = self.upscore16(upscore2 + pool4[:, :, dh:(dh + upscore2.size()[2]), dw:(dw + upscore2.size()[3])])
        # print(f'upscore16 {upscore16.size()}')


        dh, dw = (upscore16.size()[2] - input_h)//2, (upscore16.size()[3] - input_w)//2
        # print(dh, dw)
        upscore16 = upscore16[:, :, dh:(dh + input_h), dw:(dw + input_w)]
        # print(f'upscore16 {upscore16.size()}')
        return upscore16
