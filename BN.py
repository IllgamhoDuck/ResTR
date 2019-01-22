# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:02:01 2018

@author: hyunb
"""

import torch
import torch.nn as nn
from RN import RN
from TN import TN
from FC import FC
from SN import SN
from TRN import TRN

class TR(nn.Module):
    def __init__(self, args, input_size, output_size):
        super(TR, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.tn = TN(b, *input_size, 64, cuda=cuda, method=1)
        self.rn = RN(b, *input_size, 64, cuda=cuda, method=0)

        self.final = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_size)
                )

    def forward(self, x):
        tn_x = self.tn(x)
        rn_x = self.rn(x)

        output = torch.cat([tn_x, rn_x], 1)
        result = self.final(output)
        return result


class SF(nn.Module):
    def __init__(self, args, input_size, output_size):
        super(SF, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.sn = SN(b, *input_size, 64, cuda=cuda, method=1)
        self.fc = FC(*input_size, 64, cuda=cuda)

        self.final = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_size)
                )

    def forward(self, x):
        sn_x = self.sn(x)
        fc_x = self.fc(x)

        output = torch.cat([sn_x, fc_x], 1)
        result = self.final(output)
        return result
    


class TSF(nn.Module):
    def __init__(self, args, input_size, output_size):
        super(TSF, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.tn = TN(b, *input_size, 64, cuda=cuda, method=1)
        self.sn = SN(b, *input_size, 64, cuda=cuda, method=1)
        self.fc = FC(*input_size, 64, cuda=cuda)

        self.final = nn.Sequential(
                nn.Linear(192, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_size)
                )

    def forward(self, x):
        tn_x = self.tn(x)
        sn_x = self.sn(x)
        fc_x = self.fc(x)

        output = torch.cat([tn_x, sn_x, fc_x], 1)
        result = self.final(output)
        return result


class TRF(nn.Module):
    def __init__(self, args, input_size, output_size):
        super(TRF, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.tn = TN(b, *input_size, 64, cuda=cuda, method=1)
        self.rn = RN(b, *input_size, 64, cuda=cuda, method=0)
        self.fc = FC(*input_size, 64, cuda=cuda)

        self.final = nn.Sequential(
                nn.Linear(192, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, output_size)
                )

    def forward(self, x):
        tn_x = self.tn(x)
        rn_x = self.rn(x)
        fc_x = self.fc(x)

        output = torch.cat([tn_x, rn_x, fc_x], 1)
        result = self.final(output)
        return result


class TRSF(nn.Module):
    def __init__(self, args, input_size, output_size):
        super(TRSF, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.tn = TN(b, *input_size, 64, cuda=cuda, method=1)
        self.rn = RN(b, *input_size, 64, cuda=cuda, method=0)
        self.sn = SN(b, *input_size, 64, cuda=cuda, method=1)
        self.fc = FC(*input_size, 64, cuda=cuda)


        self.final = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, output_size)
                )

    def forward(self, x):
        tn_x = self.tn(x)
        rn_x = self.rn(x)
        sn_x = self.sn(x)
        fc_x = self.fc(x)

        output = torch.cat([tn_x, rn_x, sn_x, fc_x], 1)
        result = self.final(output)
        return result


class TRSF_TRN(nn.Module):
    def __init__(self, args, input_size, output_size):
        super(TRSF_TRN, self).__init__()
        
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.tn = TN(b, *input_size, 64, cuda=cuda, method=1)
        self.rn = RN(b, *input_size, 64, cuda=cuda, method=0)
        self.sn = SN(b, *input_size, 64, cuda=cuda, method=1)
        self.fc = FC(*input_size, 64, cuda=cuda)
        self.trn = TRN(b, *input_size, 64, cuda=cuda, method=0)


        self.final = nn.Sequential(
                nn.Linear(320, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, output_size)
                )
        

    def forward(self, x):
        tn_x = self.tn(x)
        rn_x = self.rn(x)
        sn_x = self.sn(x)
        fc_x = self.fc(x)
        trn_x = self.trn(x)

        output = torch.cat([tn_x, rn_x, sn_x, fc_x, trn_x], 1)
        result = self.final(output)
        return result