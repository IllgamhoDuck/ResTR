# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:02:01 2018

@author: hyunb
"""

import torch.nn as nn
from BN import TR, SF, TRF, TRSF, TRSF_TRN
from CNN import resnet50


class res_TR(nn.Module):
    def __init__(self, args):
        super(res_TR, self).__init__()
        
        self.resnet = resnet50()
        self.tr = TR(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.resnet(images)
        result = self.tr(x)
        return result
    

class res_SF(nn.Module):
    def __init__(self, args):
        super(res_SF, self).__init__()
        
        self.resnet = resnet50()
        self.sf = SF(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.resnet(images)
        result = self.sf(x)
        return result
    

class res_TRF(nn.Module):
    def __init__(self, args):
        super(res_TRF, self).__init__()
        
        self.resnet = resnet50()
        self.trf = TRF(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.resnet(images)
        result = self.trf(x)
        return result


class res_TRSF(nn.Module):
    def __init__(self, args):
        super(res_TRSF, self).__init__()
        
        self.resnet = resnet50()
        self.trsf = TRSF(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.resnet(images)
        result = self.trsf(x)
        return result


class res_TRSF_TRN(nn.Module):
    def __init__(self, args):
        super(res_TRSF_TRN, self).__init__()
        
        self.resnet = resnet50()
        self.trsf_trn = TRSF_TRN(args, (24, 8, 8), 2)
        

    def forward(self, images):
        x = self.resnet(images)
        result = self.trsf_trn(x)
        return result