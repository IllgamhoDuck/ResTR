# -*- coding: utf-8 -*-

# 206320 there is no 953645.jpg but it appeared

import configure as cf
import os
import math
import json

url = cf.url
keyword = cf.keyword
#d_type = cf.d_type

data_url = url + "/a_project/AVA_dataset/AVA.txt"
image_url = url + "/a_project/AVA_dataset/images/"

# List of the image that does not exist in real
with open(url + "/a_project/{0}_classification/dataset/del_img.json".format(keyword), "r") as f:
    del_img = json.load(f)



# AVA1 train set

# AVA1 dataset json format
# How it will be stored -> [{image: 124512, label: 0}{...}]
# 0 : bad, 1 : good
    
ava1_train_good_num = 0
ava1_train_bad_num = 0
ava1_test_good_num = 0
ava1_test_bad_num = 0  

print("####################################################")
ava1_train_url = url + "/a_project/{0}_classification/dataset/AVA1/train.txt".format(keyword) 
with open(ava1_train_url, 'r') as f:
    data = f.read()
ava1_train_img_list = data.split('\n')

ava1_train = []
missing_img_ava1_train = []
for i in range(len(ava1_train_img_list)):
    data_dict = {}
    img, label = ava1_train_img_list[i].split(' ')
    if int(label) == 0:
        ava1_train_bad_num += 1
    else:
        ava1_train_good_num +=1
    if img not in del_img:
        data_dict['image'] = img
        data_dict['label'] = int(label)
        ava1_train.append(data_dict)
    else:
        print("AVA1 train img missing! ", img)
        missing_img_ava1_train.append(img)


print("ava1 train set length is :", len(ava1_train_img_list))
print("What really saved :", len(ava1_train))
print("\n\n")

print("####################################################")
ava1_test_url = url + "/a_project/{0}_classification/dataset/AVA1/test.txt".format(keyword) 
with open(ava1_test_url, 'r') as f:
    data = f.read()
ava1_test_img_list = data.split('\n')

ava1_test = []
missing_img_ava1_test = []
for i in range(len(ava1_test_img_list)):
    data_dict = {}
    img, label = ava1_test_img_list[i].split(' ')
    if int(label) == 0:
        ava1_test_bad_num += 1
    else:
        ava1_test_good_num +=1
    if img not in del_img:
        data_dict['image'] = img
        data_dict['label'] = int(label)
        ava1_test.append(data_dict)
    else:
        print("AVA1 test img missing! ", img)
        missing_img_ava1_test.append(img)


print("ava1 test set length is :", len(ava1_test_img_list))
print("What really saved :", len(ava1_test))
print("\n")
print("ava1 train set good image number is :", ava1_train_good_num)
print("ava1 train set bad image number is :", ava1_train_bad_num)
print("\n")
print("ava1 test set good image number is :", ava1_test_good_num)
print("ava1 test set bad image number is :", ava1_test_bad_num)
print("\n")
print("ava1 good image number is :",
      ava1_train_good_num + ava1_test_good_num)
print("ava1 bad image number is :",
      ava1_train_bad_num + ava1_test_bad_num)

print("\n\n")
# AVA2 dataset json format
# How it will be stored -> [{image: 124512, label: 0}{...}]
# 0 : bad, 1 : good

ava2_train_good_num = 0
ava2_train_bad_num = 0   
ava2_test_good_num = 0
ava2_test_bad_num = 0  

print("####################################################")
ava2_train_url = url + "/a_project/{0}_classification/dataset/AVA2/train.txt".format(keyword) 
with open(ava2_train_url, 'r') as f:
    data = f.read()
ava2_train_img_list = data.split('\n')

ava2_train = []
missing_img_ava2_train = []
for i in range(len(ava2_train_img_list)):
    data_dict = {}
    img, label = ava2_train_img_list[i].split(' ')
    if int(label) == 0:
        ava2_train_bad_num += 1
    else:
        ava2_train_good_num +=1
    if img not in del_img:
        data_dict['image'] = img
        data_dict['label'] = int(label)
        ava2_train.append(data_dict)
    else:
        print("AVA2 train img missing! ", img)
        missing_img_ava2_train.append(img)


print("ava2 train set length is :", len(ava2_train_img_list))
print("What really saved :", len(ava2_train))
print("\n\n")


print("####################################################")
ava2_test_url = url + "/a_project/{0}_classification/dataset/AVA2/test.txt".format(keyword) 
with open(ava2_test_url, 'r') as f:
    data = f.read()
ava2_test_img_list = data.split('\n')

ava2_test = []
missing_img_ava2_test = []
for i in range(len(ava2_test_img_list)):
    data_dict = {}
    img, label = ava2_test_img_list[i].split(' ')
    if int(label) == 0:
        ava2_test_bad_num += 1
    else:
        ava2_test_good_num +=1
    if img not in del_img:
        data_dict['image'] = img
        data_dict['label'] = int(label)
        ava2_test.append(data_dict)
    else:
        print("AVA2 test img missing! ", img)
        missing_img_ava2_test.append(img)


print("ava2 test set length is :", len(ava2_test_img_list))
print("What really saved :", len(ava2_test))
print("\n")
print("ava2 train set good image number is :", ava2_train_good_num)
print("ava2 train set bad image number is :", ava2_train_bad_num)
print("\n")
print("ava2 test set good image number is :", ava2_test_good_num)
print("ava2 test set bad image number is :", ava2_test_bad_num)
print("\n")
print("ava2 good image number is :",
      ava2_train_good_num + ava2_test_good_num)
print("ava2 bad image number is :",
      ava2_train_bad_num + ava2_test_bad_num)

# Save files
with open(url + "/a_project/{0}_classification/dataset/{0}_ava1_train.json".format(keyword), "w") as f:
    f.write(json.dumps(ava1_train))
    
with open(url + "/a_project/{0}_classification/dataset/{0}_ava1_test.json".format(keyword), "w") as f:
    f.write(json.dumps(ava1_test))

with open(url + "/a_project/{0}_classification/dataset/{0}_ava2_train.json".format(keyword), "w") as f:
    f.write(json.dumps(ava2_train))
    
with open(url + "/a_project/{0}_classification/dataset/{0}_ava2_test.json".format(keyword), "w") as f:
    f.write(json.dumps(ava2_test))










