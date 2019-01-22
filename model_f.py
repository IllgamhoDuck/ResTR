# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:02:01 2018

@author: hyunb
"""

import torch
import torch.nn as nn
from CNN_f import cnn
from CNN_f import transfer_img
from CNN_f import resnet50_F, cnn_C
from FC import FC
from TN import TN



class dcnn_f(nn.Module):
    def __init__(self, args, fourier):
        super(dcnn_f, self).__init__()
        self.batch_size = args[0]
        self.cuda = args[1]
        self.f = fourier
        cuda = self.cuda
        
        self.conv_1 = transfer_img(fourier)
        self.conv_2 = transfer_img(fourier)
        self.cnn = cnn()
        self.cnn_f1 = cnn()
        self.cnn_f2 = cnn()
        self.fc = FC(24, 8, 8, 2, cuda)
        self.fc_f1 = FC(24, 8, 8, 64, cuda)
        self.fc_f2 = FC(24, 8, 8, 64, cuda)
        
        self.final = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 2)
                )

    def forward(self, images):
        f = self.f
        if f != 2:
            x = self.conv_1(images)
            x = self.cnn(x) # torch.Size([n, 24, 8, 8]))
            result = self.fc(x)
        else:
            x, y =self.conv_2(images)
            x = self.cnn_f1(x)
            y = self.cnn_f2(y)
            x = self.fc_f1(x)
            y = self.fc_f2(y)
            z = torch.cat([x, y], 1)
            result = self.final(z)
            
        return result



class resnet50_f(nn.Module):
    def __init__(self, args, fourier):
        super(resnet50_f, self).__init__()
        self.batch_size = args[0]
        self.cuda = args[1]
        self.f = fourier
        cuda = self.cuda
        
        self.conv_1 = transfer_img(fourier)
        self.conv_2 = transfer_img(fourier)
        self.resnet_1 = resnet50_F()
        self.resnet_2 = resnet50_F()
        self.cnn_c = cnn_C()
        self.cnn_f1 = cnn_C()
        self.cnn_f2 = cnn()
        self.fc = FC(24, 8, 8, 2, cuda)
        self.fc_f1 = FC(24, 8, 8, 64, cuda)
        self.fc_f2 = FC(24, 8, 8, 64, cuda)
        
        self.final = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 2)
                )

    def forward(self, images):
        f = self.f
        if f != 2:
            x = self.conv_1(images)
            x = self.resnet_1(x) # torch.Size([n, 2048, 7, 7])
            x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
            result = self.fc(x)
        else:
            x, y =self.conv_2(images)
            x = self.resnet_2(x)
            x = self.cnn_f1(x) 
            x = self.fc_f1(x)
            
            y = self.cnn_f2(y)
            y = self.fc_f2(y)

            z = torch.cat([x, y], 1)
            result = self.final(z)
        
        return result


class res_TN_f(nn.Module):
    def __init__(self, args, fourier):
        super(res_TN_f, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        self.f = fourier
        b = self.batch_size
        cuda = self.cuda
        
        self.conv_1 = transfer_img(fourier)
        self.conv_2 = transfer_img(fourier)
        self.resnet_1 = resnet50_F()
        self.resnet_2 = resnet50_F()
        self.cnn_c = cnn_C()
        self.cnn_c_f1 = cnn_C()
        self.cnn_f2 = cnn()
        self.tn = TN(b, 24, 8, 8, 2, cuda=cuda, method=1)
        self.tn_f1 = TN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.tn_f2 = TN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        
        self.final = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 2)
                )

    def forward(self, images):
        f = self.f
        if f != 2:
            x = self.conv_1(images)
            x = self.resnet_1(x) # torch.Size([n, 2048, 7, 7])
            x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
            result = self.tn(x)
        else:
            x, y =self.conv_2(images)
            x = self.resnet_2(x)
            x = self.cnn_c_f1(x) 
            x = self.tn_f1(x)
            
            y = self.cnn_f2(y)
            y = self.tn_f2(y)

            z = torch.cat([x, y], 1)
            result = self.final(z)
            
        return result

