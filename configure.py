# -*- coding: utf-8 -*-

# This file is made to configure every file number at one place

keyword = 'aesthetics'


# Choose the place you are training at
# AWS : 0, Own PC : 1
# AVA1 : 0, AVA2 : 1, AVA3 : 2 
# AVA3 is for test. It is consisted with only positive image

PC = 1
DATA = 1

# res + TR : 0

MODEL = 0

# Fourier transform data
FOURIER = 2

# load epoch
load_epoch_train = None
load_epoch_test = 1

categories = ['not beauty', 'beauty']
path_list = ["/jet/prs/workspace", "D:"]
data_list = ["ava1", "ava2", "ava3"]
url = path_list[PC]
d_type = data_list[DATA]
