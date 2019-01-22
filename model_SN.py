import torch
import torch.nn as nn
from SN import SN
from FC import FC
from CNN import cnn, resnet50

class cnn_SN(nn.Module):
    def __init__(self, args):
        super(cnn_SN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        # (b, 24, 8, 8)
        self.conv = cnn()
        self.sn = SN(b, 24, 8, 8, 2, cuda=cuda, method=1)
        

    def forward(self, img):
        x = self.conv(img)
        x = self.sn(x)
        
        return x

class res_SN(nn.Module):
    def __init__(self, args):
        super(res_SN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.res = resnet50()
        self.sn = SN(b, 24, 8, 8, 2, cuda=cuda, method=1)
        

    def forward(self, img):
        x = self.res(img)
        x = self.sn(x)
        
        return x


class res_SN_FC(nn.Module):
    def __init__(self, args):
        super(res_SN_FC, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.res = resnet50()
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
        x = self.res(img)
        sn_x = self.sn(x)
        fc_x = self.fc(x)
        z = torch.cat([sn_x, fc_x], 1) # (n, 128)
        
        return self.final(z)
