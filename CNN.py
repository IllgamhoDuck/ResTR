import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1) # 256 -> 128
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # 128 -> 64
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # 64 -> 32
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # 32 -> 16
        self.batchNorm4 = nn.BatchNorm2d(24)
        
        # MODIFY 14 -> 8
        self.conv5 = nn.Conv2d(24, 24, 2, stride=2, padding=1) # 16 -> 8
        self.batchNorm5 = nn.BatchNorm2d(24)
        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.batchNorm5(x)
        x = F.relu(x)
        return x


class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.conv = nn.Sequential(
                nn.Conv2d(3, 3, 33, 1, 0),
                nn.BatchNorm2d(3))

        self.pre = nn.Sequential(
                nn.Conv2d(2048, 1024, [2, 2], 1, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.Conv2d(1024, 512, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 256, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 128, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 24, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(),)
        
    def forward(self, img):
        x = self.conv(img) # 256 -> 224
        x = self.resnet(x)
        x = self.pre(x)
        return x


class transfer_img(nn.Module):
    def __init__(self, fourier):
        super(transfer_img, self).__init__()
        if fourier == 0:
            self.conv = nn.Sequential(
                    nn.Conv2d(3, 3, 33, 1, 0),
                    nn.BatchNorm2d(3))
        elif fourier == 1:
            self.conv = nn.Sequential(
                    nn.Conv2d(1, 3, 33, 1, 0),
                    nn.BatchNorm2d(3))
        
    def forward(self, images):
#        return self.conv(images)
        return images


class resnet50_F(nn.Module):
    def __init__(self):
        super(resnet50_F, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:8]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.resnet(x)
        return x


class resnet152_F(nn.Module):
    def __init__(self):
        super(resnet152_F, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.resnet(x)
        return x


class resnet50_A(nn.Module):
    def __init__(self):
        super(resnet50_A, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:5]
        self.resnet_A = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.resnet_A(x)
        return x


class resnet50_B(nn.Module):
    def __init__(self):
        super(resnet50_B, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[5]
        self.resnet_B = nn.Sequential(*modules)
        
        
    def forward(self, x):
        x = self.resnet_B(x)
        return x


class resnet50_C(nn.Module):
    def __init__(self):
        super(resnet50_C, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[6:8]
        self.resnet_C = nn.Sequential(*modules)
        
        
    def forward(self, x):
        x = self.resnet_C(x)
        return x


class cnn_A(nn.Module):
    def __init__(self):
        super(cnn_A, self).__init__()
        # torch.Size([n, 256, 56, 56])
        self.cnn_a = nn.Sequential(
                # 56 -> 28
                nn.Conv2d(256, 128, [3, 3], 2, 1, bias=False),
                nn.BatchNorm2d(128),
                # 28 -> 14
                nn.Conv2d(128, 64, [3, 3], 2, 1, bias=False),
                nn.BatchNorm2d(64),
                # 14 -> 8
                nn.Conv2d(64, 32, [2, 2], 2, 1, bias=False),
                nn.BatchNorm2d(32),
                # 8 -> 8
                nn.Conv2d(32, 24, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU())
    
    def forward(self, x):
        return self.cnn_a(x)


class cnn_B(nn.Module):
    def __init__(self):
        super(cnn_B, self).__init__()
        # torch.Size([n, 512, 28, 28])
        self.cnn_b = nn.Sequential(
                # 28 -> 28
                nn.Conv2d(512, 256, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(256),
                # 28 -> 14
                nn.Conv2d(256, 128, [3, 3], 2, 1, bias=False),
                nn.BatchNorm2d(128),
                # 14 -> 8
                nn.Conv2d(128, 64, [2, 2], 2, 1, bias=False),
                nn.BatchNorm2d(64),
                # 8 -> 8
                nn.Conv2d(64, 32, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(32),
                # 8 -> 8
                nn.Conv2d(32, 24, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU())
    
    def forward(self, x):
        return self.cnn_b(x)


class cnn_C(nn.Module):
    def __init__(self):
        super(cnn_C, self).__init__()
        # torch.Size([n, 2048, 7, 7])
        self.cnn_c = nn.Sequential(
                # 7 -> 8
                nn.Conv2d(2048, 1024, [2, 2], 1, 1, bias=False),
                nn.BatchNorm2d(1024),
                # 8 -> 8
                nn.Conv2d(1024, 512, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(512),
                # 8 -> 8
                nn.Conv2d(512, 256, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(256),
                # 8 -> 8
                nn.Conv2d(256, 128, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(128),
                # 8 -> 8
                nn.Conv2d(128, 24, [1, 1], 1, 0, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU())
    
    def forward(self, x):
        return self.cnn_c(x)