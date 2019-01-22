import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        
        # MODIFY
        self.conv5 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchNorm5(x)
        return x


class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name


class cnn(BasicModel):
    def __init__(self, args):
        super(cnn, self).__init__(args, 'cnn')
        
        self.conv = ConvInputModel()

        # MODIFY
        self.c_fc1 = nn.Linear(1536, 512)
        self.c_fc2 = nn.Linear(512, 256)
        self.c_fc3 = nn.Linear(256, 128)
        self.c_fc4 = nn.Linear(128, 2)


    def forward(self, img):
        x = self.conv(img) ## x = (64 x 24 x 8 x 8)
        x = x.view(x.size()[0], -1) # x = (64, 1536)
        
        # MODIFY
        x_ = self.c_fc1(x)
        x_ = F.relu(x_)
        x_ = self.c_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.c_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.c_fc4(x_)
        
        return x_

class cnn2(BasicModel):
    def __init__(self, args):
        super(cnn2, self).__init__(args, 'cnn2')
        
        self.conv1 = ConvInputModel()
        self.conv2 = ConvInputModel()

        # MODIFY
        self.c_fc1 = nn.Linear(3072, 512)
        self.c_fc2 = nn.Linear(512, 256)
        self.c_fc3 = nn.Linear(256, 128)
        self.c_fc4 = nn.Linear(128, 2)


    def forward(self, img):
        x = self.conv1(img) ## x = (64 x 24 x 8 x 8)
        x = x.view(x.size()[0], -1) # x = (64, 1536)
        
        y = self.conv2(img) ## x = (64 x 24 x 8 x 8)
        y = y.view(y.size()[0], -1) # x = (64, 1536)
        
        # MODIFY
        cat = torch.cat([x, y], 1)
        
        x_ = self.c_fc1(cat)
        x_ = F.relu(x_)
        x_ = self.c_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.c_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.c_fc4(x_)
        
        return x_

class cnn3(BasicModel):
    def __init__(self, args):
        super(cnn3, self).__init__(args, 'cnn3')
        
        self.conv1 = ConvInputModel()
        self.conv2 = ConvInputModel()
        self.conv3 = ConvInputModel()

        # MODIFY
        self.c_fc1 = nn.Linear(4608, 512)
        self.c_fc2 = nn.Linear(512, 256)
        self.c_fc3 = nn.Linear(256, 128)
        self.c_fc4 = nn.Linear(128, 2)


    def forward(self, img):
        x = self.conv1(img) ## x = (64 x 24 x 8 x 8)
        x = x.view(x.size()[0], -1) # x = (64, 1536)
        
        y = self.conv2(img) ## x = (64 x 24 x 8 x 8)
        y = y.view(y.size()[0], -1) # y = (64, 1536)
        
        z = self.conv3(img) ## x = (64 x 24 x 8 x 8)
        z = z.view(z.size()[0], -1) # z = (64, 1536)
        
        # MODIFY
        cat = torch.cat([x, y, z], 1)
        
        x_ = self.c_fc1(cat)
        x_ = F.relu(x_)
        x_ = self.c_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.c_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.c_fc4(x_)
        
        return x_