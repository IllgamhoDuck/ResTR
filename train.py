# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:05:46 2018

@author: hyunb
"""
import os
import json
import math
#import time 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from data_loader import get_loader
from model_test import res_TR

import configure as cf

# Hyperparameter
'''
batch_size: - duck(It means duck thinks it is useless to explain)
num_epochs: - duck(QUARK!)
save_every: determines how often to save the model weights.
            Recommend that you set save_every=1,
            to save the model weights after each epoch. 
            This way, after the ith epoch, the fashion cnn weights 
            will be saved in the models/ folder as fashion-i.pkl
print_every: determines how often to print the batch loss
log_file: the name of the text file containing - for every step
'''

# Configure file
keyword = cf.keyword
url = cf.url
d_type = cf.d_type
fourier = cf.FOURIER

batch_size = 32
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

with open(url + "/a_project/{0}_classification/dataset/{0}_{1}_train{2}.json".format(keyword, d_type, ratio), "r") as f:
    label_file = json.load(f)


data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         img_folder=img_folder,
                         label_file=label_file,
                         fourier=fourier)

args = [batch_size, torch.cuda.is_available()]
aesthetics_cnn1 = res_TR(args, fourier)


cnn_list = [aesthetics_cnn1]


aesthetics_cnn = cnn_list[cf.MODEL]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda is available? :", torch.cuda.is_available())
#aesthetics_cnn.eval()
aesthetics_cnn.to(device)

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()


params = list(aesthetics_cnn.parameters())

optimizer = optim.Adam(params, lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)



# Batch total step
total_step = math.ceil(len(data_loader.dataset) / data_loader.dataset.batch_size)
train_len = len(data_loader)

# Delete annotation if your going to load the pre-trained one

load_epoch = cf.load_epoch_train
if load_epoch:
    if fourier == 0:
        aesthetics_file = '%s-%s-%d-%d.pkl' % (keyword, d_type, cf.MODEL, load_epoch)
    else:
        aesthetics_file = '%s-%s-%d-%d_f_%d.pkl' % (keyword, d_type, cf.MODEL, load_epoch, fourier)
    if torch.cuda.is_available():
        aesthetics_cnn.load_state_dict(torch.load(os.path.join('./models', aesthetics_file)))
    else:
        aesthetics_cnn.load_state_dict(torch.load(os.path.join('./models', aesthetics_file),
                                                  map_location='cpu'))

result_file = 'result/%s-%s-%d-%d_result.txt' % (keyword, d_type, cf.MODEL, num_epochs)
learn_file = 'result/%s-%s-%d-%d_lr.txt' % (keyword, d_type, cf.MODEL, num_epochs)

# Open the training log file.
f = open(log_file, 'w')
error = open(error_file, 'w')
result = open(result_file, 'w')
learn = open(learn_file, 'w')

# Modifying when the loss will go down
#save_loss = 0.
#loss_gap = 0.025

for epoch in range(1, num_epochs+1):
    # Used only at capsule network
    epoch_acc = 0
    avg_loss = 0.
    for i_step, data in enumerate(data_loader):
        # Obtain the batch.
        # images size (batch size, 3, 224, 224)
        # labels size (batch size, 2)
        aesthetics_cnn.zero_grad()
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

        _, predicted = torch.max(output.data, 1)
        
        # Caculate the batch loss
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward() 
        
        # Update the parameters in the optimizer
        optimizer.step()
        
        # Training statistics
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % \
                (epoch, num_epochs, i_step+1, total_step, loss.item())
        result_data = '%d %d %d %d %.4f' % \
                (epoch, num_epochs, i_step+1, total_step, loss.item())
                
        # Print training statistics to file.
        f.write(stats + '\n')
        result.write(result_data + '\n')
        f.flush()
        result.flush()
        
        if (i_step+1) % print_every == 0:
            print('\r' + stats)
        
        # Save the weights.
        if (epoch % save_every == 0) and ((i_step+1) % batch_every == 0):
            print("Saved file")
            if fourier == 0:
                torch.save(aesthetics_cnn.state_dict(),
                           os.path.join('./models', '%s-%s-%d-%d.pkl' % (keyword, d_type, cf.MODEL, epoch)))
            else:
                torch.save(aesthetics_cnn.state_dict(),
                           os.path.join('./models', '%s-%s-%d-%d_f_%d.pkl' % (keyword, d_type, cf.MODEL, epoch, fourier)))
        
    
    # Save file after the atch is done
    print("Saved file")
    if fourier == 0:
        torch.save(aesthetics_cnn.state_dict(),
                   os.path.join('./models', '%s-%s-%d-%d.pkl' % (keyword, d_type, cf.MODEL, epoch)))
    else:
        torch.save(aesthetics_cnn.state_dict(),
                   os.path.join('./models', '%s-%s-%d-%d_f_%d.pkl' % (keyword, d_type, cf.MODEL, epoch, fourier)))

    scheduler.step(float(loss.item()))
    for param_groups in optimizer.param_groups:
        lr_result = 'Finished %d epoch, loss %.5f, learning rate %.10f' % (epoch, float(loss.item()), param_groups['lr'])
        print(lr_result)
        learn.write(lr_result + '\n')
        learn.flush()
            

f.close()   
error.close()
result.close()
learn.close()
