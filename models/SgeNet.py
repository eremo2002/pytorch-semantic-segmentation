import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)]
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU(inplace=False)]

            conv_bn_relu = nn.Sequential(*layers)

            return conv_bn_relu


        # Encoder
        # conv1, 224x224
        self.conv_bn_relu1_1 = conv_bn_relu(3, 64, 3, 1, 1)
        self.conv_bn_relu1_2 = conv_bn_relu(64, 64, 3, 1, 1)  

        # pool1, 112x112
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)


        # conv2, 112x112
        self.conv_bn_relu2_1 = conv_bn_relu(64, 128, 3, 1, 1)
        self.conv_bn_relu2_2 = conv_bn_relu(128, 128, 3, 1, 1)

        # pool2, 56x56
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)


        # conv3, 56x56
        self.conv_bn_relu3_1 = conv_bn_relu(128, 256, 3, 1, 1)
        self.conv_bn_relu3_2 = conv_bn_relu(256, 256, 3, 1, 1)
        self.conv_bn_relu3_3 = conv_bn_relu(256, 256, 3, 1, 1)

        # pool3, 28x28
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)


        # conv4, 28x28
        self.conv_bn_relu4_1 = conv_bn_relu(256, 512, 3, 1, 1)
        self.conv_bn_relu4_2 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv_bn_relu4_3 = conv_bn_relu(512, 512, 3, 1, 1)

        # pool4, 14x14
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        

        # conv5, 14x14
        self.conv_bn_relu5_1 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv_bn_relu5_2 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv_bn_relu5_3 = conv_bn_relu(512, 512, 3, 1, 1)

        # pool5, 7x7
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
       


        # Decoder
        # decoder5, 7x7 -> 14x14
        self.decoder_unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.decoder_conv_bn_relu5_1 = conv_bn_relu(512, 512, 3, 1, 1)
        self.decoder_conv_bn_relu5_2 = conv_bn_relu(512, 512, 3, 1, 1)
        self.decoder_conv_bn_relu5_3 = conv_bn_relu(512, 512, 3, 1, 1)


        # decoder4, 14x14 -> 28x28
        self.decoder_unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.decoder_conv_bn_relu4_1 = conv_bn_relu(512, 512, 3, 1, 1)
        self.decoder_conv_bn_relu4_2 = conv_bn_relu(512, 512, 3, 1, 1)
        self.decoder_conv_bn_relu4_3 = conv_bn_relu(512, 256, 3, 1, 1)


        # decoder3, 28x28 -> 56x56
        self.decoder_unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.decoder_conv_bn_relu3_1 = conv_bn_relu(256, 256, 3, 1, 1)
        self.decoder_conv_bn_relu3_2 = conv_bn_relu(256, 256, 3, 1, 1)
        self.decoder_conv_bn_relu3_3 = conv_bn_relu(256, 128, 3, 1, 1)


        # decoder2, 56x56 -> 112x112
        self.decoder_unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.decoder_conv_bn_relu2_1 = conv_bn_relu(128, 128, 3, 1, 1)
        self.decoder_conv_bn_relu2_2 = conv_bn_relu(128, 64, 3, 1, 1)        


        # decoder1, 112x112 -> 224x224
        self.decoder_unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.decoder_conv_bn_relu1_1 = conv_bn_relu(64, 64, 3, 1, 1)
        self.decoder_conv_bn_relu1_2 = conv_bn_relu(64, 64, 3, 1, 1)
        
        self.score_fr = nn.Conv2d(64, num_classes, 1)       
        

    def forward(self, x):
        # Encoder
        x = self.conv_bn_relu1_1(x)
        x = self.conv_bn_relu1_2(x)
        encoder_size1 = x.size()
        x, pool1_indices = self.pool1(x)

        x = self.conv_bn_relu2_1(x)
        x = self.conv_bn_relu2_2(x)
        encoder_size2 = x.size()
        x, pool2_indices = self.pool2(x)

        x = self.conv_bn_relu3_1(x)
        x = self.conv_bn_relu3_2(x)
        x = self.conv_bn_relu3_3(x)
        encoder_size3 = x.size()
        x, pool3_indices = self.pool3(x)

        x = self.conv_bn_relu4_1(x)
        x = self.conv_bn_relu4_2(x)
        x = self.conv_bn_relu4_3(x)
        encoder_size4 = x.size()
        x, pool4_indices = self.pool4(x)

        x = self.conv_bn_relu5_1(x)
        x = self.conv_bn_relu5_2(x)
        x = self.conv_bn_relu5_3(x)
        encoder_size5 = x.size()
        x, pool5_indices = self.pool5(x)


        # Decoder
        x = self.decoder_unpool5(x, pool5_indices, output_size=encoder_size5)
        x = self.decoder_conv_bn_relu5_1(x)
        x = self.decoder_conv_bn_relu5_2(x)
        x = self.decoder_conv_bn_relu5_3(x)

        x = self.decoder_unpool4(x, pool4_indices, output_size=encoder_size4)
        x = self.decoder_conv_bn_relu4_1(x)
        x = self.decoder_conv_bn_relu4_2(x)
        x = self.decoder_conv_bn_relu4_3(x)

        x = self.decoder_unpool3(x, pool3_indices, output_size=encoder_size3)
        x = self.decoder_conv_bn_relu3_1(x)
        x = self.decoder_conv_bn_relu3_2(x)
        x = self.decoder_conv_bn_relu3_3(x)

        x = self.decoder_unpool2(x, pool2_indices, output_size=encoder_size2)
        x = self.decoder_conv_bn_relu2_1(x)
        x = self.decoder_conv_bn_relu2_2(x)

        x = self.decoder_unpool1(x, pool1_indices, output_size=encoder_size1)
        x = self.decoder_conv_bn_relu1_1(x)
        x = self.decoder_conv_bn_relu1_2(x)

        x = self.score_fr(x)

        return x        
