import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class RN(nn.Module):
    def __init__(self, batch_size, channel, width, height,
                 output_size, cuda=False, method=0):
        super(RN, self).__init__()
    
        # Check is cuda true
        self.cuda = cuda
        
        # what size to output
        self.output = output_size
        
        # CNN Feature box        
        # (n x channel x width x height)
        self.batch_size = batch_size
        self.channel = channel
        self.width = width
        self.height = height
        
        # How to process the g part
        # 0, use g_0. After that sum all by row and go to f
        # 1, use g_1. After that change the view to [b,w*h]
        self.method = method
        
        # Initialize the batch size coordinate tensor
        self.coord_tensor = self.create_coord(batch_size)
        
        
        # method 0 g,f
        self.g_0 = nn.Sequential(
                nn.Linear((channel+2)*2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU())
        
        self.f_0 = nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, output_size))
        
        # method 1 g,f
        self.g_1 = nn.Sequential(
                nn.Linear((channel+2)*2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 2),
                nn.BatchNorm1d(2),
                nn.ReLU())
        
        self.f_1 = nn.Sequential(
                nn.Linear(width*height*width*height*2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, output_size))
        

    def cvt_coord(self, i):
        middle = self.height // 2
        return [(i/self.height-middle)/float(middle),
                (i%self.height-middle)/float(middle)]
    
    def create_coord(self, batch_size):
        b = batch_size
        w = self.width
        h = self.height
        
        # Create coordinate tensor
        coord_tensor = torch.FloatTensor(b,w*h,2)
        if self.cuda:
            coord_tensor = coord_tensor.cuda()
        coord_tensor = Variable(coord_tensor)
        
        # Fill using numpy array
        np_coord_tensor = np.zeros((b,w*h,2))
        for i in range(w*h):
            np_coord_tensor[:,i,:] = np.array(self.cvt_coord(i))
        coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        
        return coord_tensor
    
    def forward(self, cnn_tensor):
        # Check the batch size because it can change at the very last batch
        # 1000 images, batch size 64
        # The last tensor size is 40
        batch_size = cnn_tensor.size()[0]
        c = self.channel
        w = self.width
        h = self.height
        
        # flat tensor
        # (b,c,w,h) -> (b,c,w*h) -> (b,w*h,c)
        cnn_flat = cnn_tensor.view(batch_size,c,w*h).permute(0,2,1)
        
        # add coordinates
        # (b,w*h,c) -> (b,w*h,c+2)
        if batch_size == self.batch_size:
            cnn_dot = torch.cat([cnn_flat, self.coord_tensor], 2)
        else:
            coord_tensor = self.create_coord(batch_size)
            cnn_dot = torch.cat([cnn_flat, coord_tensor], 2)
        
        # cast all pairs against each other
        cnn_dot_i = torch.unsqueeze(cnn_dot, 1) # (b,1,w*h,c+2)
        cnn_dot_i = cnn_dot_i.repeat(1,w*h,1,1) # (b,w*h,w*h,c+2)
        cnn_dot_j = torch.unsqueeze(cnn_dot, 2) # (b,w*h,1,c+2)
        cnn_dot_j = cnn_dot_j.repeat(1,1,w*h,1) # (b,w*h,w*h,c+2)
        
        # Concatenate cnn dot & cnn total
        # (b,w*h,w*h,(c+2)*2)
        cnn_full = torch.cat([cnn_dot_i, cnn_dot_j], 3)
        
        # (b,w*h,w*h,(c+2)*2) -> (b*w*h*w*h,(c+2)*2)
        cnn_full = cnn_full.view(batch_size*w*h*w*h,(c+2)*2)
        
        if self.method == 0:
            # g_0
            # (b,w*h,c) -> (b,256)
            cnn_g_0 = self.g_0(cnn_full)
            cnn_g_0 = cnn_g_0.view(batch_size,w*h*w*h,256)
            cnn_g_0 = cnn_g_0.sum(1).squeeze()
            
            # f_0
            # (b,256) -> (b,output_size)
            result = self.f_0(cnn_g_0)
        elif self.method == 1:
            # g_1
            # (b,w*h,c) -> (b,w*h*w*h*2)
            cnn_g_1 = self.g_1(cnn_full)
            cnn_g_1 = cnn_g_1.view(batch_size,w*h*w*h*2)
            
            # f_1
            # (b,w*h*w*h*2) -> (b,output_size)
            result = self.f_1(cnn_g_1)
        else:
            raise Exception('Choose the method 0 or 1')
        
        return result
    
