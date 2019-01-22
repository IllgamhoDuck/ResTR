import torch
import torch.nn as nn
from TN import TN
from FC import FC
from CNN import cnn, resnet50

class cnn_TN(nn.Module):
    def __init__(self, args):
        super(cnn_TN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        # (b, 24, 8, 8)
        self.conv = cnn()
        self.tn = TN(b, 24, 8, 8, 2, cuda=cuda, method=1)
        

    def forward(self, img):
        x = self.conv(img)
        x = self.tn(x)
        
        return x

class res_TN(nn.Module):
    def __init__(self, args):
        super(res_TN, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.res = resnet50()
        self.tn = TN(b, 24, 8, 8, 2, cuda=cuda, method=1)
        

    def forward(self, img):
        x = self.res(img)
        x = self.tn(x)
        
        return x


class res_TN_FC(nn.Module):
    def __init__(self, args):
        super(res_TN_FC, self).__init__()

        self.batch_size = args[0]
        self.cuda = args[1]
        b = self.batch_size
        cuda = self.cuda
        
        self.res = resnet50()
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
        x = self.res(img)
        tn_x = self.tn(x)
        fc_x = self.fc(x)
        z = torch.cat([tn_x, fc_x], 1) # (n, 128)
        
        return self.final(z)
