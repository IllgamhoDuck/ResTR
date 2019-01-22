# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:02:01 2018

@author: hyunb
"""

import torch
import torch.nn as nn
from CNN import transfer_img
from CNN import resnet50_A, cnn_A
from CNN import resnet50_B, cnn_B
from CNN import resnet50_C, cnn_C
from BN import TRF, TRSF, TRSF_TRN
from FC import FC
from TN import TN


class beauty_net(nn.Module):
    def __init__(self, args, fourier):
        super(beauty_net, self).__init__()
        cuda = args[1]
        
        self.conv = transfer_img(fourier)
        self.resnet_a = resnet50_A()
        self.cnn_a = cnn_A()
        
        self.resnet_b = resnet50_B()
        self.cnn_b = cnn_B()
        
        self.resnet_c = resnet50_C()
        self.cnn_c = cnn_C()
        self.fc_1 = FC(24, 8, 8, 64, cuda)
        self.fc_2 = FC(24, 8, 8, 64, cuda)
        self.fc_3 = FC(24, 8, 8, 64, cuda)

       
        self.final = nn.Sequential(
                nn.Linear(192, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 2)
                )

    def forward(self, images):
        
        # Local
        x = self.conv(images) # torch.Size([n, 3, 224, 224])
        a_x = self.resnet_a(x) # torch.Size([n, 256, 56, 56])
        cnn_a_x = self.cnn_a(a_x) # torch.Size([n, 24, 8, 8])
        a_final = self.fc_1(cnn_a_x) # (n, 64)
        
        # Middle
        b_x = self.resnet_b(a_x) # torch.Size([n, 512, 28, 28])
        cnn_b_x = self.cnn_b(b_x) # torch.Size([n, 24, 8, 8]))
        b_final = self.fc_2(cnn_b_x) # (n, 64)
        
        # Global
        c_x = self.resnet_c(b_x) # torch.Size([n, 2048, 7, 7])
        cnn_c_x = self.cnn_c(c_x) # torch.Size([n, 24, 8, 8]))
        c_final = self.fc_3(cnn_c_x) # (n, 64)
        
        final = torch.cat([a_final, b_final, c_final], 1)
        result = self.final(final)
        
        return result


class beauty_TN(nn.Module):
    def __init__(self, args, fourier):
        super(beauty_TN, self).__init__()
        b = args[0]
        cuda = args[1]
        
        self.conv = transfer_img(fourier)
        self.resnet_a = resnet50_A()
        self.cnn_a = cnn_A()
        
        self.resnet_b = resnet50_B()
        self.cnn_b = cnn_B()
        
        self.resnet_c = resnet50_C()
        self.cnn_c = cnn_C()
        self.tn_1 = TN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.tn_2 = TN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.tn_3 = TN(b, 24, 8, 8, 64, cuda=cuda, method=1)

       
        self.final = nn.Sequential(
                nn.Linear(192, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 2)
                )

    def forward(self, images):
        
        # Local
        x = self.conv(images) # torch.Size([n, 3, 224, 224])
        a_x = self.resnet_a(x) # torch.Size([n, 256, 56, 56])
        cnn_a_x = self.cnn_a(a_x) # torch.Size([n, 24, 8, 8])
        a_final = self.tn_1(cnn_a_x) # (n, 64)
        
        # Middle
        b_x = self.resnet_b(a_x) # torch.Size([n, 512, 28, 28])
        cnn_b_x = self.cnn_b(b_x) # torch.Size([n, 24, 8, 8]))
        b_final = self.tn_2(cnn_b_x) # (n, 64)
        
        # Global
        c_x = self.resnet_c(b_x) # torch.Size([n, 2048, 7, 7])
        cnn_c_x = self.cnn_c(c_x) # torch.Size([n, 24, 8, 8]))
        c_final = self.tn_3(cnn_c_x) # (n, 64)
        
        final = torch.cat([a_final, b_final, c_final], 1)
        result = self.final(final)
        
        return result