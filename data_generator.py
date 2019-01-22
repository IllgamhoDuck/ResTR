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

# MAKING THE DATA FILE THAT MATCHES THE REAL EXISTING FILE!!!
#==============================================================
# Check the image list at AVA dataset
# real image list
r_img_list = os.listdir(image_url)
r_set = set(r_img_list)

with open(data_url, 'r') as f:
    data = f.read()

data = data.split('\n')
del data[-1]

# AVA.txt image list
a_img_list = []
for i in range(len(data)):
    a_img_list.append(str(data[i].split(" ")[1]) + '.jpg') 
    data[i] = [int(i) for i in data[i].split(" ")]

a_set = set(a_img_list)
# 'final.log', no problem with this!
diff_r = r_set.difference(a_set)
# 22 images
diff_a = a_set.difference(r_set)
a_del = list(diff_a)

# Save the image that does not exist
with open(url + "/a_project/{0}_classification/dataset/del_img.json".format(keyword), "w") as f:
    f.write(json.dumps(a_del))

# Make a list to delete in the ava dataset file
index_del = []
for i in range(len(a_del)):
    index_del.append(a_img_list.index(a_del[i]))
    
# sort the index_del to descending order
index_del.sort(reverse=True)

# Delete the image in the data list
for i in range(len(index_del)):
    del data[index_del[i]]

# Checking if there is no error!
test_img_list = []
for i in range(len(data)):
    test_img_list.append(str(data[i][1]) + '.jpg')
    
test_set = set(test_img_list)
diff_r = r_set.difference(test_set)
diff_test = test_set.difference(r_set)

assert len(list(diff_test)) == 0
#==============================================================


# The function calculates the mean score of image
def cal_mean_score(score_list):
    total_score = 0
    total_vote = 0
    for i in range(10):
        score = i + 1
        vote_num = score_list[i + 2]
        total_vote += vote_num
        total_score += score * vote_num
    return round(total_score / total_vote, 3)


# Evaluate the mean score of each data
# data[i][1] - ID
# data[i][2] ~ data[i][11] - The number of vote of score 1 to 10
img_list = []
for i in range(len(data)):
    image_score = []
    mean_score = cal_mean_score(data[i])
    image_score.append(data[i][1])
    image_score.append(mean_score)
    img_list.append(image_score)
    
img_list.sort(key = lambda x:x[1])


# AVA1 dataset
# Split the dataset good if mean score bigger than 5
# And bad if mean score is lower than 5
bound_num = None
for i in range(len(img_list)):
    if math.floor(img_list[i][1]) == 5:
        bound_num = i
        break

ava1_bad_img_list = img_list[:bound_num]
ava1_good_img_list = img_list[bound_num:]

print("AVA1 Dataset lenght of bad : ",
      len(ava1_bad_img_list))
print("AVA1 Dataset lenght of good : ",
      len(ava1_good_img_list))
print(ava1_bad_img_list[-1])
print(ava1_good_img_list[0])
print('\n')

    
# Consider top 10% and bottom 10% 
percent_num = math.ceil(len(img_list) / 10)
ava2_bad_img_list = img_list[:percent_num]
ava2_good_img_list = img_list[-percent_num:]

print("AVA2 Dataset lenght of bad : ",
      len(ava2_bad_img_list))
print("AVA2 Dataset lenght of good : ",
      len(ava2_good_img_list))
print(ava2_bad_img_list[-1])
print(ava2_good_img_list[0])


# AVA1 dataset json format
# How it will be stored -> [{image: 124512, label: 0}{...}]
# 0 : bad, 1 : good
ava1_good = []
for i in range(len(ava1_good_img_list)):
    label = {}
    label['image'] = str(ava1_good_img_list[i][0]) + ".jpg"
    label['label'] = 1
    ava1_good.append(label)

ava1_bad = []
for i in range(len(ava1_bad_img_list)):
    label = {}
    label['image'] = str(ava1_bad_img_list[i][0]) + ".jpg"
    label['label'] = 0
    ava1_bad.append(label)


# AVA2 dataset json format
# How it will be stored -> [{image: 124512, label: 0}{...}]
# 0 : bad, 1 : good
ava2_good = []
for i in range(len(ava2_good_img_list)):
    label = {}
    label['image'] = str(ava2_good_img_list[i][0]) + ".jpg"
    label['label'] = 1
    ava2_good.append(label)

ava2_bad = []
for i in range(len(ava2_bad_img_list)):
    label = {}
    label['image'] = str(ava2_bad_img_list[i][0]) + ".jpg"
    label['label'] = 0
    ava2_bad.append(label)

#==============================================================
# SET THE RATE
#==============================================================
# To compare with other paper need to split this to 9:1
divide_rate = 0.9



# AVA1 train, test dataset
# Split this to train 80%, test 20% or etc
ava1_train = []
ava1_test = []

total_good_len = len(ava1_good)
good_train_len = math.ceil(divide_rate * total_good_len)
good_test_len = total_good_len - good_train_len

ava1_train.extend(ava1_good[:good_train_len])
ava1_test.extend(ava1_good[good_train_len:])

total_bad_len = len(ava1_bad)
bad_train_len = math.ceil(divide_rate * total_bad_len)
bad_test_len = total_bad_len - bad_train_len

ava1_train.extend(ava1_bad[:bad_train_len])
ava1_test.extend(ava1_bad[bad_train_len:])

print("AVA1 train set length : ", good_train_len + bad_train_len)
print("AVA1 test set length : ", good_test_len + bad_test_len)
print('\n')


# AVA2 train, test dataset
# Split this to train 80%, test 20% or etc
ava2_train = []
ava2_test = []

total_good_len = len(ava2_good)
good_train_len = math.ceil(divide_rate * total_good_len)
good_test_len = total_good_len - good_train_len

ava2_train.extend(ava2_good[:good_train_len])
ava2_test.extend(ava2_good[good_train_len:])

total_bad_len = len(ava2_bad)
bad_train_len = math.ceil(divide_rate * total_bad_len)
bad_test_len = total_bad_len - bad_train_len

ava2_train.extend(ava2_bad[:bad_train_len])
ava2_test.extend(ava2_bad[bad_train_len:])

print("AVA2 train set length : ", good_train_len + bad_train_len)
print("AVA2 test set length : ", good_test_len + bad_test_len)


# Save the dataset file
#with open(url + "/a_project/{0}_classification/{0}_{1}_train.json".format(keyword, d_type), "w") as f:
#    f.write(json.dumps(ava1_train))
#    
#with open(url + "/a_project/{0}_classification/{0}_{1}_test.json".format(keyword, d_type), "w") as f:
#    f.write(json.dumps(ava1_test))

if divide_rate == 0.8:
    with open(url + "/a_project/{0}_classification/dataset/{0}_ava1_train_82.json".format(keyword), "w") as f:
        f.write(json.dumps(ava1_train))
        
    with open(url + "/a_project/{0}_classification/dataset/{0}_ava1_test_82.json".format(keyword), "w") as f:
        f.write(json.dumps(ava1_test))
    
    with open(url + "/a_project/{0}_classification/dataset/{0}_ava2_train_82.json".format(keyword), "w") as f:
        f.write(json.dumps(ava2_train))
        
    with open(url + "/a_project/{0}_classification/dataset/{0}_ava2_test_82.json".format(keyword), "w") as f:
        f.write(json.dumps(ava2_test))

if divide_rate == 0.9:
    with open(url + "/a_project/{0}_classification/dataset/{0}_ava1_train_91.json".format(keyword), "w") as f:
        f.write(json.dumps(ava1_train))
        
    with open(url + "/a_project/{0}_classification/dataset/{0}_ava1_test_91.json".format(keyword), "w") as f:
        f.write(json.dumps(ava1_test))
    
    with open(url + "/a_project/{0}_classification/dataset/{0}_ava2_train_91.json".format(keyword), "w") as f:
        f.write(json.dumps(ava2_train))
        
    with open(url + "/a_project/{0}_classification/dataset/{0}_ava2_test_91.json".format(keyword), "w") as f:
        f.write(json.dumps(ava2_test))

# Make a test file that has only one image.
# This is used to see if the model is training or not

# Only positive image
#positive = []
#for i in range(1000):
#    positive.append(ava1_train[0])
#with open(url + "/a_project/{0}_classification/{0}_ava3_train.json".format(keyword), "w") as f:
#    f.write(json.dumps(positive))
    
# Only negative image
#negative = []
#for i in range(1000):
#    negative.append(ava1_train[201121])
#with open(url + "/a_project/{0}_classification/{0}_ava3_test.json".format(keyword), "w") as f:
#    f.write(json.dumps(negative))








