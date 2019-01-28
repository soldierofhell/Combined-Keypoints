#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 14:54:54 2017

@author: dinesh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
#from image_plot import *
import sys,os


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

def write_data(file,data):
    file.write(str(data[0]))
    for index,value in enumerate(data):
        if index == 0:
            continue
        file.write(',')
        file.write(str(data[index]))
    file.write('\n')

def write_file(filename,data):
    file = open(filename,'w') 
    for ind,kp in enumerate(data):
        write_data(file,data[ind])
    file.close()


Folder = sys.argv[1]#'/home/dinesh/CarCrash/data/Fifth/'
folder_save = sys.argv[3]#'/home/dinesh/CarCrash/data/Fifth/'
#Folder = '/home/dinesh/CarCrash/data/CarCrash/Cleaned/'
#Folder = '/home/dinesh/CarCrash/data/syn/'
#Folder = '/home/dinesh/CarCrash/data/Kitti_1/'
#Folder = '/home/dinesh/CarCrash/data/test/'



main_loop = int(sys.argv[2])

# front head lights
#for main_loop in range(1,21):
filenames_delete = glob.glob(Folder + str(main_loop) + '/' + folder_save + '/*_*')
#filenames_delete = glob.glob(Folder + str(main_loop) + '/keypoints_txt_new/*_*')
print(filenames_delete)
for index,del_name in enumerate(filenames_delete):
    os.remove(del_name)

filenames = sorted(glob.glob(Folder + str(main_loop) + '/' + folder_save +'/*.txt'),key=lambda x: int(x.split('/')[-1].split('.')[0]))
unique_tracks = 0
for index,name in enumerate(filenames):
    bb = []
    points = []
    class_name = []
    img_name = name.split(folder_save)[0] + name.split(folder_save)[1].split('.txt')[0]
    img_original = cv2.imread(img_name)
    data = []
    with open(filenames[index]) as f:
        lines = f.readlines()
    for line in lines:
        #print(line)
        data.append(line.split('\n')[0].split(','))

    if index ==0:
        write_file(name.replace(folder_save,folder_save+ 'tracked'),data)
        unique_tracks += len(data)
        data_prev = data
        continue
        
    data_prev = []
    filename = filenames[index-1]
    filename = filename.replace(folder_save,folder_save + 'tracked')
    print(filename)
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        data_prev.append(line.split('\n')[0].split(','))
        
    #file = open(Folder + '/Car3D/' + str(time).zfill(5) +  '.txt','w') 
    for ind,kp in enumerate(data):
        flag = True
        for ind_new,kp_new in enumerate(data_prev):
            BOXA = [int(float(l)) for l in kp_new[1:5]]
            BOXB = [int(float(l)) for l in kp[1:5]]
                
            if _compute_iou(BOXA,BOXB) > 0.7:
                #print(data[ind][0],data_prev[ind_new][0])
                data[ind][0] = data_prev[ind_new][0]
                flag = False
                break
        if flag == True:
                
            unique_tracks += 1
            data[ind][0] = str(unique_tracks)
                
        
    write_file(name.replace(folder_save,folder_save+ 'tracked'),data)
        #print(data)     
        #print(data_prev)     
        #asas
    #img_instance_segment = cv2.imread(img_name.replace('//','/labelled/'))
    data_prev = data
        #adsas
        
