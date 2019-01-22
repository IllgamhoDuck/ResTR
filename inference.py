# -*- coding: utf-8 -*-

import os
import math
import json
import numpy as np

from data_loader import get_loader
from model_test import res_TR

import torch
import torch.nn as nn
from torchvision import transforms

import configure as cf

transform_test = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
# Configure file
keyword = cf.keyword
url = cf.url
d_type = cf.d_type
fourier = cf.FOURIER

categories = cf.categories
error_file = 'error_log_inference.txt'

batch_size = 5

# TODO - Change the images file to the folder where the image is
img_folder = url + "/a_project/AVA_dataset/images/"


# TODO - Change the label file where is filled with 
#        [{'image':image_id(str),'label':category index(int)},{},...,{}]

label_file = []
                     
with open(url + "/a_project/{0}_classification/dataset/{0}_{1}_test{2}.json".format(keyword, d_type, ratio), "r") as f:
    label_file = json.load(f)

category_len = len(categories)

data_loader = get_loader(transform=transform_test,
                         mode='test',
                         batch_size=batch_size,
                         img_folder=img_folder,
                         label_file=label_file,
                         fourier=fourier)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda is available? :", torch.cuda.is_available())

args = [batch_size, torch.cuda.is_available()]
aesthetics_cnn1 = res_TR(args, fourier)


cnn_list = [aesthetics_cnn1]

aesthetics_cnn = cnn_list[cf.MODEL]

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()


# Load the pre-trained one
load_epoch = cf.load_epoch_test
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

aesthetics_cnn.eval()
aesthetics_cnn.to(device)

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
test_loss = torch.zeros(1)
class_correct = list(0. for i in range(category_len))
class_total = list(0. for i in range(category_len))

f = open('{0}_{1}_test_result.json'.format(keyword, d_type), 'w')
error = open(error_file, 'w')

result_list = []
index = 0

# Batch total step
total_step = math.ceil(len(data_loader.dataset) / data_loader.dataset.batch_size)
test_len = len(data_loader)
acc = 0

for i_step, data in enumerate(data_loader):
    # Obtain the batch.
    # images size (batch size, 3, 224, 224)
    # labels size (batch size, 5)
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
    
    # Calculate the batch loss
    loss = criterion(output, labels)
    
    # Update average test loss
    test_loss = test_loss + ((torch.ones(1) / (i_step + 1)) * (loss.item() - test_loss))
    
    # get the predicted class from the maximum value in the output_list of class scores
    # (batch size, 1)
    _, predicted = torch.max(output.data, 1)

#    print('label and predicted\n')
#    print(labels.cpu().numpy())
#    print(predicted.cpu().numpy())
#
    pre_list = list(predicted.cpu().numpy())
    lab_list = list(labels.cpu().numpy())
    for i in range(len(pre_list)):
        pre_dict = {}
        pre_dict['index'] = index
        pre_dict['label'] = categories[lab_list[i]]
        pre_dict['predict'] = categories[pre_list[i]]
        index += 1
        result_list.append(pre_dict)
    
    # compare predictions to true label
    # correct = [0,1,1,...,1,0,1]
    correct = 0
    correct += (predicted == labels)
    if (i_step + 1) % 100 == 0:
        print('\nbatch step: ', i_step+1)
        print('test_loss: ', round(test_loss.item(), 3))
    
    # calculate test accuracy for each object class
    for i in range(len(labels)):
        label = labels.data[i].item()
        class_correct[label] += correct[i].item()
        class_total[label] += 1
    if (i_step + 1) % 100 == 0:
        print('Test Accuracy : %.2f%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        print('class correct :', class_correct)
        print('class total :', class_total)


print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

for i in range(category_len):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %.2f%% (%2d/%2d)' % (
                categories[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy  of %5s: N/A (no training examples)' % (categories[i]))

print('\nTest Accuracy (Overall): %.2f%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

f.write(json.dumps(result_list))
print("Result written in result file")
