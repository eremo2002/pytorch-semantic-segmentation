import torch
import torch.nn as nn
import torch.nn.functional as F

class DeconvNet(nn.Module):
    def __init__(self, num_classes):
        super(DeconvNet, self).__init__()

        # Conv Part

        # conv1, 224x224
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=False)

        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=False)


        # pool1, 224x224 -> 112x112
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)


        # conv2, 112x112
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=False)  

        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=False)        
        
        
        # pool2, 112x112 -> 56x56
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)


        # conv3, 56x56
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=False)

        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=False)

        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=False)
        

        # pool3, 56x56 -> 28x28
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)


        # conv4, 28x28
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=False)
        
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=False)

        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=False)
        

        # pool4, 28x28 -> 14x14
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)


        # conv5, 14x14
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=False)

        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=False)

        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=False)
        

        # pool5, 14x14 -> 7x7
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True)


        # fc6, 7x7 -> 1x1
        self.fc6_1 = nn.Conv2d(512, 4096, 7)
        self.bn6_1 = nn.BatchNorm2d(4096)
        self.relu6_1 = nn.ReLU(inplace=False)        


        # fc7, 1x1
        self.fc7_1 = nn.Conv2d(4096, 4096, 1)
        self.bn7_1 = nn.BatchNorm2d(4096)
        self.relu7_1 = nn.ReLU(inplace=False)


        # Deconv pat

        # fc7_deconv, 1x1 -> 7x7
        self.fc7_deconv = nn.ConvTranspose2d(4096, 512, 7)
        self.bn_fc7 = nn.BatchNorm2d(512)
        self.relu_fc7 = nn.ReLU(inplace=False)

        
        # unpool5, 7x7 -> 14x14
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        
        # deconv5, 14x14
        self.deconv5_1 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.debn5_1 = nn.BatchNorm2d(512)
        self.derelu5_1 = nn.ReLU(inplace=False)  

        self.deconv5_2 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.debn5_2 = nn.BatchNorm2d(512)
        self.derelu5_2 = nn.ReLU(inplace=False)  

        self.deconv5_3 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.debn5_3 = nn.BatchNorm2d(512)
        self.derelu5_3 = nn.ReLU(inplace=False)  

        
        # unpool4, 14x14 -> 28x28
        self.unpool4 = nn.MaxUnpool2d(2, 2)


        # deconv4, 28x28
        self.deconv4_1 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.debn4_1 = nn.BatchNorm2d(512)
        self.derelu4_1 = nn.ReLU(inplace=False)  

        self.deconv4_2 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.debn4_2 = nn.BatchNorm2d(512)
        self.derelu4_2 = nn.ReLU(inplace=False)  

        self.deconv4_3 = nn.ConvTranspose2d(512, 256, 3, padding=1)
        self.debn4_3 = nn.BatchNorm2d(256)
        self.derelu4_3 = nn.ReLU(inplace=False)  


        # unpool3, 28x28 -> 56x56
        self.unpool3 = nn.MaxUnpool2d(2, 2)


        # deconv3, 56x56
        self.deconv3_1 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.debn3_1 = nn.BatchNorm2d(256)
        self.derelu3_1 = nn.ReLU(inplace=False)  

        self.deconv3_2 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.debn3_2 = nn.BatchNorm2d(256)
        self.derelu3_2 = nn.ReLU(inplace=False)  

        self.deconv3_3 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.debn3_3 = nn.BatchNorm2d(128)
        self.derelu3_3 = nn.ReLU(inplace=False)  


        # unpool2, 56x56 -> 112x112
        self.unpool2 = nn.MaxUnpool2d(2, 2)


        # deconv2, 112x112
        self.deconv2_1 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.debn2_1 = nn.BatchNorm2d(128)
        self.derelu2_1 = nn.ReLU(inplace=False)  

        self.deconv2_2 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.debn2_2 = nn.BatchNorm2d(64)
        self.derelu2_2 = nn.ReLU(inplace=False)  


        # unpool1, 112x112 -> 224x224
        self.unpool1 = nn.MaxUnpool2d(2, 2)


        # deconv1, 224x224
        self.deconv1_1 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.debn1_1 = nn.BatchNorm2d(64)
        self.derelu1_1 = nn.ReLU(inplace=False)  

        self.deconv1_2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.debn1_2 = nn.BatchNorm2d(64)
        self.derelu1_2 = nn.ReLU(inplace=False) 

        # seg-score
        self.score = nn.Conv2d(64, num_classes, 1)
        


        
        

    def forward(self, x):
        # conv1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)  

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)  

        conv1_size = x.size()

        # pool1
        x, pool1_ind = self.pool1(x)


        # conv2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)  

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)  

        conv2_size = x.size()

        # pool2
        x, pool2_ind = self.pool2(x)


        # conv3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)  

        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)  

        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu3_3(x)  

        conv3_size = x.size()

        # pool3
        x, pool3_ind = self.pool3(x)


        # conv4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)  

        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu4_2(x)  

        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.relu4_3(x)  

        conv4_size = x.size()

        # pool4
        x, pool4_ind = self.pool4(x)


        # conv5
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu5_1(x)  

        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu5_2(x)  

        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.relu5_3(x)  

        conv5_size = x.size()

        # pool5        
        x, pool5_ind = self.pool5(x)


        # fc6
        x = self.fc6_1(x)
        x = self.bn6_1(x)
        x = self.relu6_1(x)


        # fc7
        x = self.fc7_1(x)
        x = self.bn7_1(x)
        x = self.relu7_1(x)

        
        # Deconv part
        # fc7_deconv, 7x7
        x = self.fc7_deconv(x)
        x = self.bn_fc7(x)
        x = self.relu_fc7(x)


        # unpool5
        x = self.unpool5(x, pool5_ind, conv5_size)


        # deconv5
        x = self.deconv5_1(x)
        x = self.debn5_1(x)
        x = self.derelu5_1(x)
        
        x = self.deconv5_2(x)
        x = self.debn5_2(x)
        x = self.derelu5_2(x)

        x = self.deconv5_3(x)
        x = self.debn5_3(x)
        x = self.derelu5_3(x)


        # unpool4
        x = self.unpool4(x, pool4_ind, conv4_size)


        # deconv4
        x = self.deconv4_1(x)
        x = self.debn4_1(x)
        x = self.derelu4_1(x)
        
        x = self.deconv4_2(x)
        x = self.debn4_2(x)
        x = self.derelu4_2(x)

        x = self.deconv4_3(x)
        x = self.debn4_3(x)
        x = self.derelu4_3(x)


        # unpool3
        x = self.unpool3(x, pool3_ind, conv3_size)       


        # deconv3
        x = self.deconv3_1(x)
        x = self.debn3_1(x)
        x = self.derelu3_1(x)
        
        x = self.deconv3_2(x)
        x = self.debn3_2(x)
        x = self.derelu3_2(x)

        x = self.deconv3_3(x)
        x = self.debn3_3(x)
        x = self.derelu3_3(x)


        # unpool2
        x = self.unpool2(x, pool2_ind, conv2_size)


        # deconv2
        x = self.deconv2_1(x)
        x = self.debn2_1(x)
        x = self.derelu2_1(x)
        
        x = self.deconv2_2(x)
        x = self.debn2_2(x)
        x = self.derelu2_2(x)


        # unpool1
        x = self.unpool1(x, pool1_ind, conv1_size)


        # deconv1
        x = self.deconv1_1(x)
        x = self.debn1_1(x)
        x = self.derelu1_1(x)
        
        x = self.deconv1_2(x)
        x = self.debn1_2(x)
        x = self.derelu1_2(x)
        
        
        # seg-score
        x = self.score(x)        
        return x

