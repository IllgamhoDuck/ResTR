import torch
import torch.nn as nn
from RN import RN
from FC import FC
from TRN import TRN
from CNN import cnn, resnet50


class cnn_RN(nn.Module):
    def __init__(self, args):
        super(cnn_RN, self).__init__()
        
        self.conv = cnn()
        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.rn = RN(b, 24, 8, 8, 2, cuda=cuda, method=0)
        

    def forward(self, img):
        x = self.conv(img) ## x = (64 x 24 x 8 x 8)
        x = self.rn(x)
        
        return x

class res_RN(nn.Module):
    def __init__(self, args):
        super(res_RN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.res = resnet50()
        self.rn = RN(b, 24, 8, 8, 2, cuda=cuda, method=0)
        

    def forward(self, img):
        x = self.res(img)
        x = self.rn(x)
        
        return x


class res_TRN(nn.Module):
    def __init__(self, args):
        super(res_TRN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.res = resnet50()
        self.trn = TRN(b, 24, 8, 8, 2, cuda=cuda, method=0)
        

    def forward(self, img):
        x = self.res(img)
        x = self.trn(x)
        
        return x


class res_RN_FC(nn.Module):
    def __init__(self, args):
        super(res_RN_FC, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.res = resnet50()
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
        x = self.res(img)
        rn_x = self.rn(x)
        fc_x = self.fc(x)
        z = torch.cat([rn_x, fc_x], 1) # (n, 128)
        
        return self.final(z)