# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:02:01 2018

@author: hyunb
"""

import torch
import torch.nn as nn
from CNN import transfer_img
from CNN import resnet50_F, resnet152_F, cnn_C
from BN import TR, SF
from FC import FC
from TN import TN
from RN import RN
from TRN import TRN
from CNN import cnn



class dcnn(nn.Module):
    def __init__(self, args, fourier):
        super(dcnn, self).__init__()
        self.batch_size = args[0]
        self.cuda = args[1]
        cuda = self.cuda
        
        self.conv = transfer_img(fourier)
        self.cnn = cnn()
        self.fc = FC(24, 8, 8, 2, cuda)

    def forward(self, images):
        
        x = self.conv(images)
        x = self.cnn(x) # torch.Size([n, 24, 8, 8]))
        x = self.fc(x)
        
        return x


class resnet50(nn.Module):
    def __init__(self, args, fourier):
        super(resnet50, self).__init__()
        self.batch_size = args[0]
        self.cuda = args[1]
        cuda = self.cuda
        
        self.conv = transfer_img(fourier)
        self.resnet = resnet50_F()
        self.cnn_c = cnn_C()
        self.fc = FC(24, 8, 8, 2, cuda)

    def forward(self, images):
        
        x = self.conv(images)
        x = self.resnet(images) # torch.Size([n, 2048, 7, 7])
        x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
        x = self.fc(x)
        
        return x


class resnet152(nn.Module):
    def __init__(self, args, fourier):
        super(resnet152, self).__init__()
        self.batch_size = args[0]
        self.cuda = args[1]
        cuda = self.cuda
        
        self.conv = transfer_img(fourier)
        self.resnet = resnet152_F()
        self.cnn_c = cnn_C()
        self.fc = FC(24, 8, 8, 2, cuda)

    def forward(self, images):
        
        x = self.conv(images)
        x = self.resnet(images) # torch.Size([n, 2048, 7, 7])
        x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
        x = self.fc(x)
        
        return x


class res_TN(nn.Module):
    def __init__(self, args, fourier):
        super(res_TN, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.conv = transfer_img(fourier)
        self.resnet = resnet50_F()
        self.cnn_c = cnn_C()
        self.tn = TN(b, 24, 8, 8, 2, cuda=cuda, method=1)

    def forward(self, images):
        
        x = self.conv(images)
        x = self.resnet(images) # torch.Size([n, 2048, 7, 7])
        x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
        x = self.tn(x)
        
        return x


class res_TR(nn.Module):
    def __init__(self, args, fourier):
        super(res_TR, self).__init__()
        
        self.conv = transfer_img(fourier)
        self.resnet = resnet50_F()
        self.cnn_c = cnn_C()
        self.tr = TR(args, (24, 8, 8), 2)

    def forward(self, images):
        
        x = self.conv(images)
        x = self.resnet(images) # torch.Size([n, 2048, 7, 7])
        x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
        x = self.tr(x)
        
        return x


class res_SF(nn.Module):
    def __init__(self, args, fourier):
        super(res_SF, self).__init__()
        
        self.conv = transfer_img(fourier)
        self.resnet = resnet50_F()
        self.cnn_c = cnn_C()
        self.sf = SF(args, (24, 8, 8), 2)

    def forward(self, images):
        
        x = self.conv(images)
        x = self.resnet(images) # torch.Size([n, 2048, 7, 7])
        x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
        x = self.sf(x)
        
        return x
    


class res_RN(nn.Module):
    def __init__(self, args, fourier):
        super(res_RN, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.conv = transfer_img(fourier)
        self.resnet = resnet50_F()
        self.cnn_c = cnn_C()
        self.rn = RN(b, 24, 8, 8, 2, cuda=cuda, method=0)

    def forward(self, images):
        
        x = self.conv(images)
        x = self.resnet(images) # torch.Size([n, 2048, 7, 7])
        x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
        x = self.rn(x)
        
        return x


class res_TRN(nn.Module):
    def __init__(self, args, fourier):
        super(res_TRN, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.conv = transfer_img(fourier)
        self.resnet = resnet50_F()
        self.cnn_c = cnn_C()
        self.trn = TRN(b, 24, 8, 8, 2, cuda=cuda, method=0)

    def forward(self, images):
        
        x = self.conv(images)
        x = self.resnet(images) # torch.Size([n, 2048, 7, 7])
        x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
        x = self.trn(x)
        
        return x


class res_test(nn.Module):
    def __init__(self, args, fourier):
        super(res_test, self).__init__()
        
        self.conv = transfer_img(fourier)
        self.resnet = resnet50_F()
        self.tr = TR(args, (24, 8, 8), 64)
        self.sf = SF(args, (24, 8, 8), 64)
        self.cnn_c = cnn_C()

        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, images):
        
        x = self.conv(images)
        x = self.resnet(images) # torch.Size([n, 2048, 7, 7])
        x = self.cnn_c(x) # torch.Size([n, 24, 8, 8]))
        
        tr_x = self.tr(x)
        sf_x = self.sf(x)
        z = torch.cat([tr_x, sf_x], 1) # (n, 128)
        
        return self.final(z)


class cnn_test(nn.Module):
    def __init__(self, args):
        super(cnn_test, self).__init__()

        self.cnn = cnn()
        self.tr = TR(args, (24, 8, 8), 64)
        self.sf = SF(args, (24, 8, 8), 64)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn(img)
        tr_x = self.tr(x)
        sf_x = self.sf(x)
        z = torch.cat([tr_x, sf_x], 1) # (n, 128)
        
        return self.final(z)
    

class cnn_test2(nn.Module):
    def __init__(self, args):
        super(cnn_test2, self).__init__()

        self.cnn_1 = cnn()
        self.cnn_2 = cnn()
        self.tr_1 = TR(args, (24, 8, 8), 64)
        self.sf_1 = SF(args, (24, 8, 8), 64)
        self.tr_2 = TR(args, (24, 8, 8), 64)
        self.sf_2 = SF(args, (24, 8, 8), 64)
        self.af_1 = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
                )
        self.af_2 = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
                )
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn_1(img)
        y = self.cnn_2(img)
        
        tr_x = self.tr_2(x)
        sf_x = self.sf_2(x)
        
        tr_y = self.tr_2(y)
        sf_y = self.sf_2(y)
        
        af_x = torch.cat([tr_x, sf_x], 1) # (n, 128)
        af_y = torch.cat([tr_y, sf_y], 1) # (n, 128)
        
        z_1 = self.af_1(af_x)
        z_2 = self.af_2(af_y)
        
        z = torch.cat([z_1, z_2], 1)
        
        return self.final(z)


class cnn_SF2(nn.Module):
    def __init__(self, args):
        super(cnn_SF2, self).__init__()

        self.cnn_1 = cnn()
        self.cnn_2 = cnn()
        self.sf_1 = SF(args, (24, 8, 8), 64)
        self.sf_2 = SF(args, (24, 8, 8), 64)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn_1(img)
        y = self.cnn_2(img)
        sf_x = self.sf_1(x)
        sf_y = self.sf_2(y)
        z = torch.cat([sf_x, sf_y], 1) # (n, 128)
        
        return self.final(z)
    

class cnn_TR2(nn.Module):
    def __init__(self, args):
        super(cnn_TR2, self).__init__()

        self.cnn_1 = cnn()
        self.cnn_2 = cnn()
        self.tr_1 = TR(args, (24, 8, 8), 64)
        self.tr_2 = TR(args, (24, 8, 8), 64)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn_1(img)
        y = self.cnn_2(img)
        tr_x = self.tr_1(x)
        tr_y = self.tr_2(y)
        z = torch.cat([tr_x, tr_y], 1) # (n, 128)
        
        return self.final(z)


class cnn_TRSF(nn.Module):
    def __init__(self, args):
        super(cnn_TRSF, self).__init__()

        self.cnn_1 = cnn()
        self.cnn_2 = cnn()
        self.tr = TR(args, (24, 8, 8), 64)
        self.sf = SF(args, (24, 8, 8), 64)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn_1(img)
        y = self.cnn_2(img)
        tr_x = self.tr(x)
        sf_y = self.sf(y)
        z = torch.cat([tr_x, sf_y], 1) # (n, 128)
        
        return self.final(z)
