# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:05:46 2018

@author: hyunb
"""
import json
import math
#import time 

from data_loader import get_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from model_f import dcnn_f, resnet50_f, res_TN_f
from model_test import res_RN, res_TRN, res_SF


import configure as cf

# Configure file
keyword = cf.keyword
url = cf.url
d_type = cf.d_type
fourier = cf.FOURIER

batch_size = 2
num_epochs = 10
save_every = 1
batch_every = 100
print_every = 10
log_file = 'training_log.txt'
error_file = 'error_log_train.txt'

transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# TODO - Change the images file to the folder where the image is
img_folder = url + "/a_project/AVA_dataset/images/"

# TODO - Change the label file where is filled with 
#        [{'image':image_id(str),'label':category index(int)},{},...,{}]
label_file = []

with open(url + "/a_project/{0}_classification/dataset/{0}_{1}_train.json".format(keyword, d_type), "r") as f:
    label_file = json.load(f)


data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         img_folder=img_folder,
                         label_file=label_file,
                         fourier=fourier)

args = [batch_size, torch.cuda.is_available()]
aesthetics_cnn = dcnn_f(args, fourier)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda is available? :", torch.cuda.is_available())
#aesthetics_cnn.eval()
aesthetics_cnn.to(device)


criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
params = list(aesthetics_cnn.parameters())
optimizer = optim.Adam(params, lr=0.0001)



# Batch total step
total_step = math.ceil(len(data_loader.dataset) / data_loader.dataset.batch_size)
train_len = len(data_loader)

for epoch in range(1, num_epochs+1):
    for i_step, data in enumerate(data_loader):
        if fourier != 2:
            images, labels = data
            
            images = images.to(device)
            labels = labels.to(device)

            output = aesthetics_cnn(images)
        else:
            images, images_f, labels = data

            images = images.to(device)
            images_f = images_f.to(device)
            labels = labels.to(device)
            
            output = aesthetics_cnn([images, images_f])
        print(output.size())

            
    
