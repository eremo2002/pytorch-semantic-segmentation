import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias)]
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU(inplace=False)]

            conv_bn_relu = nn.Sequential(*layers)

            return conv_bn_relu


        # Contracting path        
        self.enc1_1 = conv_bn_relu(1, 64, 3, 1, 0, True)
        self.enc1_2 = conv_bn_relu(64, 64, 3, 1, 0, True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.enc2_1 = conv_bn_relu(64, 128, 3, 1, 0, True)
        self.enc2_2 = conv_bn_relu(128, 128, 3, 1, 0, True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.enc3_1 = conv_bn_relu(128, 256, 3, 1, 0, True)
        self.enc3_2 = conv_bn_relu(256, 256, 3, 1, 0, True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.enc4_1 = conv_bn_relu(256, 512, 3, 1, 0, True)
        self.enc4_2 = conv_bn_relu(512, 512, 3, 1, 0, True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.enc5_1 = conv_bn_relu(512, 1024, 3, 1, 0, True)


        # Expanding path 
        self.dec5_1 = conv_bn_relu(1024, 1024, 3, 1, 0, True)
          
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = conv_bn_relu(1024, 512, 3, 1, 0, True)
        self.dec4_1 = conv_bn_relu(512, 512, 3, 1, 0, True)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2 = conv_bn_relu(512, 256, 3, 1, 0, True)
        self.dec3_1 = conv_bn_relu(256, 256, 3, 1, 0, True)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = conv_bn_relu(256, 128, 3, 1, 0, True)
        self.dec2_1 = conv_bn_relu(128, 128, 3, 1, 0, True)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = conv_bn_relu(128, 64, 3, 1, 0, True)
        self.dec1_1 = conv_bn_relu(64, 64, 3, 1, 0, True)

        self.score = nn.Conv2d(64, num_classes, 1)
        

    def forward(self, x):
        # Contracting path
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)  

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2) 

        enc5_1 = self.enc5_1(pool4)     
        #print(enc1_2.size(), enc2_2.size(), enc3_2.size(), enc4_2.size())
        

        # Expanding path   
        dec5_1 = self.dec5_1(enc5_1)
        
        upconv4 = self.upconv4(dec5_1)        
        # center crop for concat because two Tensors must be same size
        diff_h, diff_w = abs(upconv4.size()[2] - enc4_2.size()[2])//2, abs(upconv4.size()[3] - enc4_2.size()[3])//2
        enc4_2 = enc4_2[:, :, diff_h:(diff_h + upconv4.size()[2]), diff_w:(diff_w + upconv4.size()[3])]                

        cat4 = torch.cat((upconv4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)


        upconv3 = self.upconv3(dec4_1)
        diff_h, diff_w = abs(upconv3.size()[2] - enc3_2.size()[2])//2, abs(upconv3.size()[3] - enc3_2.size()[3])//2
        enc3_2 = enc3_2[:, :, diff_h:(diff_h + upconv3.size()[2]), diff_w:(diff_w + upconv3.size()[3])]        
        cat3 = torch.cat((upconv3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)


        upconv2 = self.upconv2(dec3_1)
        diff_h, diff_w = abs(upconv2.size()[2] - enc2_2.size()[2])//2, abs(upconv2.size()[3] - enc2_2.size()[3])//2
        enc2_2 = enc2_2[:, :, diff_h:(diff_h + upconv2.size()[2]), diff_w:(diff_w + upconv2.size()[3])]        
        cat2 = torch.cat((upconv2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)


        upconv1 = self.upconv1(dec2_1)
        diff_h, diff_w = abs(upconv1.size()[2] - enc1_2.size()[2])//2, abs(upconv1.size()[3] - enc1_2.size()[3])//2
        enc1_2 = enc1_2[:, :, diff_h:(diff_h + upconv1.size()[2]), diff_w:(diff_w + upconv1.size()[3])]
        cat1 = torch.cat((upconv1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.score(dec1_1)        
       
        return output
