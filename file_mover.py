# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:16:18 2018

@author: hyunbyung87
"""

import os
import json
import shutil

import configure as cf

url = cf.url
keyword = cf.keyword

image_url = url + "/a_project/AVA_dataset/images/"
save_url = url + "/a_project/image_check/ava2/"

########################
# Open AVA1 good & bad #
########################
#ava1_good_list = None
#ava1_bad_list = None
#
#with open(url + "/a_project/{0}_classification/dataset/{0}_ava1_good.json".format(keyword), "r") as f:
#        ava1_good_list = json.load(f)
#        
#with open(url + "/a_project/{0}_classification/dataset/{0}_ava1_bad.json".format(keyword), "r") as f:
#        ava1_bad_list = json.load(f)

########################
# Open AVA2 good & bad #
########################
ava2_good_list = None
ava2_bad_list = None

with open(url + "/a_project/{0}_classification/dataset/{0}_ava2_good.json".format(keyword), "r") as f:
        ava2_good_list = json.load(f)
        
with open(url + "/a_project/{0}_classification/dataset/{0}_ava2_bad.json".format(keyword), "r") as f:
        ava2_bad_list = json.load(f)

good_url = os.path.join(save_url, 'good')
bad_url = os.path.join(save_url, 'bad')

# AVA2 GOOD
before_copy = len(os.listdir(good_url))
for label in ava2_good_list:
    image = label['image']
    src = os.path.join(image_url, image)
    dst = good_url
    shutil.copy2(src, dst)
after_copy = len(os.listdir(good_url))
print("AVA2 Good image copied total number is : {}\n\n".format(after_copy - before_copy))

# AVA2 BAD
before_copy = len(os.listdir(bad_url))
for label in ava2_bad_list:
    image = label['image']
    src = os.path.join(image_url, image)
    dst = bad_url
    shutil.copy2(src, dst)
after_copy = len(os.listdir(bad_url))
print("AVA2 Bad image copied total number is : {}\n\n".format(after_copy - before_copy))