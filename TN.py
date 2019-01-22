import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class TN(nn.Module):
    def __init__(self, batch_size, channel, width, height,
                 output_size, total_vector_size=None,
                 cuda=False, method=0):
        super(TN, self).__init__()
    
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
        
        # output size of total vector
        if total_vector_size == None:
            self.tv_size = channel
        else:
            self.tv_size = total_vector_size
        
        # How to process the g part
        # 0, use g_0. After that sum all by row and go to f
        # 1, use g_1. After that change the view to [b,w*h]
        self.method = method
        
        # Initialize the batch size coordinate tensor
        self.coord_tensor = self.create_coord(batch_size)
        
        # Create fc layer to make TOTAL VIEW TENSOR!!!
        self.fc = nn.Sequential(
                nn.Linear(width*height*channel, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, self.tv_size))
        
        # method 0 g,f
        self.g_0 = nn.Sequential(
                nn.Linear(channel+2+self.tv_size, 256),
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
                nn.Linear(channel+2+self.tv_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, channel),
                nn.BatchNorm1d(channel),
                nn.ReLU())
        
        self.f_1 = nn.Sequential(
                nn.Linear(width*height*channel, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
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
        
        tv_size = self.tv_size
        
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
        
        # Make the tensor that represents total view
        # (b,c,w,h) -> (b,c*w*h)
        cnn_total = cnn_tensor.view(batch_size, -1)
        
        # (b,c*w*h) -> (b,tv_size)
        cnn_total = self.fc(cnn_total)
        
        # (b,tv_size) -> (b,1,tv_size) -> (b,w*h,tv_size)
        cnn_total = torch.unsqueeze(cnn_total, 1)
        cnn_total = cnn_total.repeat(1,w*h,1)
        
        # Concatenate cnn dot & cnn total
        # (b,w*h,c+2+tv_size)
        cnn_full = torch.cat([cnn_dot, cnn_total], 2)
        
        # (b,w*h,c+2+tv_size) -> (b*w*h,c+2+tv_size)
        cnn_full = cnn_full.view(batch_size*w*h,c+2+tv_size)
        
        if self.method == 0:
            # g_0
            # (b*w*h,c+2+tv_size) -> (b,256)
            cnn_g_0 = self.g_0(cnn_full)
            cnn_g_0 = cnn_g_0.view(batch_size,w*h,256)
            cnn_g_0 = cnn_g_0.sum(1).squeeze()
            
            # f_0
            # (b,256) -> (b,output_size)
            result = self.f_0(cnn_g_0)
        elif self.method == 1:
            # g_1
            # (b*w*h,c+2+tv_size) -> (b,w*h*10)
            cnn_g_1 = self.g_1(cnn_full)
            cnn_g_1 = cnn_g_1.view(batch_size,w*h*c)
            
            # f_1
            # (b,w*h*10) -> (b,output_size)
            result = self.f_1(cnn_g_1)
        else:
            raise Exception('Choose the method 0 or 1')
        
        return result
    
