# -*- coding: utf-8 -*-

import os
import numpy as np

from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_loader(transform,
               mode='train',
               batch_size=1,
               img_folder="/",
               label_file=[],
               num_workers=0,
               fourier=0):
    """
    Args:
        transform : Image tranform
        mode: One of 'train' or 'test', this arguments are not used at here
        batch_size: Batch Size
        img_folder: the folder path that images are stored
        label_file: The file that stores annotation about the image and label
        num_workers: Number of Subprocesses to use data loading
    
    [Detailed explantion about the label file]
    =========================================================================
    Annotation label_file data structure
    
    list(dict)
    [{'image': image_id(str), 'label', label(int)}]
    
    Ex: [{'image': 'im_1' 'label': 1},
         {'image': 'im_2' 'label': 3},
         {'image': 'im_3' 'label': 7}]
    
    Image id is another expression of image path.
    The loader will load the image by it's path.
    
    label number starts from 0.
    For this label it has 18 different categories.
    So it will start at 0 and end at 17.
    =========================================================================
    
    There is no need to distinguish the train and the test.
    If your going to split to train and test image,
    
    Split the label_file to for example 8:2
    """
    
    assert len(label_file) != 0, "Please fill the label file QUARK!"
    if fourier == 0:
        dataset = aesthetics_dataset(transform=transform,
                                  mode=mode,
                                  batch_size=batch_size,
                                  label_file=label_file,
                                  img_folder=img_folder)
    elif fourier == 1:
        dataset = aesthetics_dataset_f(transform=transform,
                                  mode=mode,
                                  batch_size=batch_size,
                                  label_file=label_file,
                                  img_folder=img_folder)
    elif fourier == 2:
        dataset = aesthetics_dataset_f2(transform=transform,
                                  mode=mode,
                                  batch_size=batch_size,
                                  label_file=label_file,
                                  img_folder=img_folder)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    return data_loader

class aesthetics_dataset(data.Dataset):
    '''
    Dataset to train the clothes category classification model
    
    The __getitem__ will return 2 variables label and image.

    image = numpy array with size (3, 224, 224)    
    label = Number label file with size (1)  
    '''
    def __init__(self, transform, mode, batch_size, label_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.label_file = label_file
        self.img_folder = img_folder
    
    def __getitem__(self, index):
        label_num = self.label_file[index]['label']
        label = torch.tensor(label_num)
        
        #open image
        img_path = os.path.join(self.img_folder,
                                self.label_file[index]['image'])
        PIL_image = Image.open(img_path).convert('RGB')
        orig_image = PIL_image
        image = self.transform(orig_image)
    
        return image, label
    
    def __len__(self):
        return len(self.label_file)


class aesthetics_dataset_f(data.Dataset):
    '''
    Dataset to train the clothes category classification model
    
    The __getitem__ will return 2 variables label and image.

    image = numpy array with size (3, 224, 224)    
    label = Number label file with size (1)  
    '''
    def __init__(self, transform, mode, batch_size, label_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.label_file = label_file
        self.img_folder = img_folder
        self.transform_1 = transforms.Compose([
                transforms.Resize(256),                         
                transforms.RandomCrop(224)])
        self.transform_2 = transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
    
    def __getitem__(self, index):
        label_num = self.label_file[index]['label']
        label = torch.tensor(label_num)
        
        #open image
        img_path = os.path.join(self.img_folder,
                                self.label_file[index]['image'])
        PIL_image = Image.open(img_path).convert('RGB')
        PIL_image = self.transform_1(PIL_image)
        grey_im = PIL_image.convert('L')
        im = np.array(grey_im)
        
        # Fast Fourier Transform
        im = np.fft.fft2(im)
        im = np.fft.fftshift(im)
        im = np.log(np.abs(im+1))
        im = im / 255.0
        
        im = np.array(im, np.float32, copy=False)
        im = torch.from_numpy(im)
        im = im.unsqueeze(0)
        image = self.transform_2(im)
    
        return image, label
    
    def __len__(self):
        return len(self.label_file)


class aesthetics_dataset_f2(data.Dataset):
    '''
    Dataset to train the clothes category classification model
    
    The __getitem__ will return 2 variables label and image.

    image = numpy array with size (3, 224, 224)    
    label = Number label file with size (1)  
    '''
    def __init__(self, transform, mode, batch_size, label_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.label_file = label_file
        self.img_folder = img_folder
        self.transform_1 = transforms.Compose([
                transforms.Resize(256),                         
                transforms.RandomCrop(224)])
        self.transform_2 = transforms.Compose([
                transforms.ToTensor(),                         
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.transform_3 = transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
    
    def __getitem__(self, index):
        label_num = self.label_file[index]['label']
        label = torch.tensor(label_num)
        
        #open image
        img_path = os.path.join(self.img_folder,
                                self.label_file[index]['image'])
        PIL_image = Image.open(img_path).convert('RGB')
        PIL_image = self.transform_1(PIL_image)
        orig_im = np.array(PIL_image)
        image = self.transform_2(PIL_image)
        
        grey_im = PIL_image.convert('L')
        im = np.array(grey_im)

        # Fast Fourier Transform
        im = np.fft.fft2(im)
        im = np.fft.fftshift(im)
        im = np.log(np.abs(im+1))
        im = im / 255.0
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        print("This image is : ", "GOOD" if label_num == 1 else "BAD")
        ax1.imshow(orig_im)
        ax2.imshow(im, cmap='gray')
        
        im = np.array(im, np.float32, copy=False)
        im = torch.from_numpy(im)
        im = im.unsqueeze(0)
        image_f = self.transform_3(im)
    
        return image, image_f, label
    
    def __len__(self):
        return len(self.label_file)