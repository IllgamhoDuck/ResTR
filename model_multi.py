# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:02:01 2018

@author: hyunb
"""

import torch
import torch.nn as nn
from RN import RN
from TN import TN
from SN import SN
from FC import FC
from BN import TSF, TRF, TRSF, TRSF_TRN
from CNN import cnn


class cnn_SN_FC(nn.Module):
    def __init__(self, args):
        super(cnn_SN_FC, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.cnn = cnn()
        self.sn = SN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.fc = FC(24, 8, 8, 64, cuda)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn(img)
        sn_x = self.sn(x)
        fc_x = self.fc(x)
        z = torch.cat([sn_x, fc_x], 1) # (n, 128)
        
        return self.final(z)


class cnn_TN_FC(nn.Module):
    def __init__(self, args):
        super(cnn_TN_FC, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.cnn = cnn()
        self.tn = TN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.fc = FC(24, 8, 8, 64, cuda)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn(img)
        tn_x = self.tn(x)
        fc_x = self.fc(x)
        z = torch.cat([tn_x, fc_x], 1) # (n, 128)
        
        return self.final(z)


class cnn_RN_FC(nn.Module):
    def __init__(self, args):
        super(cnn_RN_FC, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.cnn = cnn()
        self.rn = RN(b, 24, 8, 8, 64, cuda=cuda, method=0)
        self.fc = FC(24, 8, 8, 64, cuda)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn(img)
        rn_x = self.rn(x)
        fc_x = self.fc(x)
        z = torch.cat([rn_x, fc_x], 1) # (n, 128)
        
        return self.final(z)


class cnn_TN_RN(nn.Module):
    def __init__(self, args):
        super(cnn_TN_RN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.cnn = cnn()
        self.tn = TN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.rn = RN(b, 24, 8, 8, 64, cuda=cuda, method=0)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn(img)
        tn_x = self.tn(x)
        rn_x = self.rn(x)
        z = torch.cat([tn_x, rn_x], 1) # (n, 128)
        
        return self.final(z)
        

class cnn_TN_SN(nn.Module):
    def __init__(self, args):
        super(cnn_TN_SN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.cnn = cnn()
        self.tn = TN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.sn = SN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn(img)
        tn_x = self.tn(x)
        sn_x = self.sn(x)
        z = torch.cat([tn_x, sn_x], 1) # (n, 128)
        
        return self.final(z)


# Understanding the dots and the relation between the dots
class cnn_RN_SN(nn.Module):
    def __init__(self, args):
        super(cnn_RN_SN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.cnn = cnn()
        self.rn = RN(b, 24, 8, 8, 64, cuda=cuda, method=0)
        self.sn = SN(b, 24, 8, 8, 64, cuda=cuda, method=1)
        self.final = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
                )

    def forward(self, img):
        x = self.cnn(img)
        rn_x = self.rn(x)
        sn_x = self.sn(x)
        z = torch.cat([rn_x, sn_x], 1) # (n, 128)
        
        return self.final(z)


# Understanding the dots and total view and the relation between them
class cnn_TSF(nn.Module):
    def __init__(self, args):
        super(cnn_TSF, self).__init__()
        
        self.cnn = cnn()
        self.tsf = TSF(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.cnn(images)
        result = self.tsf(x)
        return result


class cnn_TRF(nn.Module):
    def __init__(self, args):
        super(cnn_TRF, self).__init__()
        
        self.cnn = cnn()
        self.trf = TRF(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.cnn(images)
        result = self.trf(x)
        return result


class cnn_TRSF(nn.Module):
    def __init__(self, args):
        super(cnn_TRSF, self).__init__()
        
        self.cnn = cnn()
        self.trsf = TRSF(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.cnn(images)
        result = self.trsf(x)
        return result


class cnn_TRSF_TRN(nn.Module):
    def __init__(self, args):
        super(cnn_TRSF_TRN, self).__init__()
        
        self.cnn = cnn()
        self.trsf_trn = TRSF_TRN(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.cnn(images)
        result = self.trsf_trn(x)
        return result