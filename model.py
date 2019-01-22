# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:02:01 2018

@author: hyunb
"""

import torch
import torch.nn as nn
import torchvision.models as models
import configure as cf

from CNN import cnn_C, transfer_img
from FC import FC

fourier = cf.FOURIER

#class resnet152(nn.Module):
#    def __init__(self, args, fourier):
#        super(resnet152, self).__init__()
#        resnet = models.resnet152(pretrained=True)
#        modules = list(resnet.children())[:-2]
#        
#        self.batch_size = args[0]
#        self.cuda = args[1]
#        cuda = self.cuda
#
#        self.conv = transfer_img(fourier)
#        self.resnet = nn.Sequential(*modules)
#        self.cnn = cnn_C()
#        self.fc = FC(24, 8, 8, 2, cuda)
#
#    def forward(self, images):
#        x = self.conv(images)
#        x = self.resnet(x)
#        x = self.cnn(x)
#        x = self.fc(x)
#        return x


class resnet152(nn.Module):
    def __init__(self, args, fourier):
        super(resnet152, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Sequential(
                nn.Linear(resnet.fc.in_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 2),
                )
        
        if fourier == 0:
            self.conv = nn.Sequential(
                    nn.Conv2d(3, 3, 33, 1, 0),
                    nn.BatchNorm2d(3),
                    nn.ReLU())
        elif fourier == 1:
            self.conv = nn.Sequential(
                    nn.Conv2d(1, 3, 33, 1, 0),
                    nn.BatchNorm2d(3),
                    nn.ReLU())

    def forward(self, images):
        output = self.conv(images)
        output = self.resnet(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output



class res_aesthetic_CNN(nn.Module):
    def __init__(self):
        super(res_aesthetic_CNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
#        for param in resnet.parameters():
#            param.requires_grad_(False)
        
        # Requires the last CNN layer to be trained
        # This requires more computing power
#        self.last_children = list(resnet.children())[-3]
#        self.last_bottle = list(self.last_children)[-1]
#        for param in self.last_bottle.parameters():
#            param.requires_grad_(True)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Sequential(
                nn.Linear(resnet.fc.in_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 2),
                )
        
        if fourier == 0:
            self.conv = nn.Sequential(
                    nn.Conv2d(3, 3, 33, 1, 0),
                    nn.BatchNorm2d(3),
                    nn.ReLU())
        elif fourier == 1:
            self.conv = nn.Sequential(
                    nn.Conv2d(1, 3, 33, 1, 0),
                    nn.BatchNorm2d(3),
                    nn.ReLU())

    def forward(self, images):
        output = self.conv(images)
        output = self.resnet(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output



class res_aesthetic_CNN_2(nn.Module):
    def __init__(self):
        super(res_aesthetic_CNN_2, self).__init__()
        resnet_1 = models.resnet50(pretrained=True)
        resnet_2 = models.resnet50(pretrained=True)
        
        modules_1 = list(resnet_1.children())[:-1]
        modules_2 = list(resnet_2.children())[:-1]

        self.resnet_1 = nn.Sequential(*modules_1)
        self.resnet_2 = nn.Sequential(*modules_2)

        self.fc = nn.Sequential(
                nn.Linear(resnet_1.fc.in_features * 2, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.BatchNorm1d(256),
                nn.Linear(256, 2),
                )
        
        if fourier == 0:
            self.conv = nn.Sequential(
                    nn.Conv2d(3, 3, 33, 1, 0),
                    nn.BatchNorm2d(3),
                    nn.ReLU())
        elif fourier == 1:
            self.conv = nn.Sequential(
                    nn.Conv2d(1, 3, 33, 1, 0),
                    nn.BatchNorm2d(3),
                    nn.ReLU())

    def forward(self, images):
        output = self.conv(images)
        output_1 = self.resnet_1(output)
        output_2 = self.resnet_2(output)
        output_1 = output_1.view(output_1.size(0), -1)
        output_2 = output_2.view(output_2.size(0), -1)

        output = torch.cat([output_1, output_2], 1)
        output = self.fc(output)
        return output
