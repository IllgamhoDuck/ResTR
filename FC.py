import torch.nn as nn


class FC(nn.Module):
    def __init__(self, channel, width, height, output_size, cuda):
        super(FC, self).__init__()
    
        # Check is cuda true
        self.cuda = cuda
        
        # what size to output
        self.output = output_size
        
        # CNN Feature box        
        # (n x channel x width x height)
        self.channel = channel
        self.width = width
        self.height = height
        
        
        # Create fc layer
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
                nn.Linear(256, output_size))
        
    
    def forward(self, cnn_tensor):
        # Check the batch size because it can change at the very last batch
        # 1000 images, batch size 64
        # The last tensor size is 40
        batch_size = cnn_tensor.size()[0]
        
        # FC
        # (b,c,w,h) -> (b,c*w*h)
        cnn_total = cnn_tensor.view(batch_size, -1)
        result = self.fc(cnn_total)
        
        return result
    
