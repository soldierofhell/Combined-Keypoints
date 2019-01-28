from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer


import itertools
import matplotlib.pyplot as plt

import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

save_image = False
def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg_car', dest='cfg_file_car',
        help='optional config file')
    parser.add_argument(
        '--cfg_person', dest='cfg_file_person',
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt_car', help='path of checkpoint to load')
    parser.add_argument('--load_ckpt_person', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)
    parser.add_argument(
        '--visualize', type=distutils.util.strtobool, default=False)


    args = parser.parse_args()

    return args

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def distance(a,b):
	return np.linalg.norm(a-b)
def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset.startswith("keypoints_carfusion"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))




    assert bool(args.load_ckpt_car) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
 


    print('load cfg from file: {}'.format(args.cfg_file_person))
    cfg_from_file(args.cfg_file_person)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.RESNETS.IMAGENET_PRETRAINED = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN_person = Generalized_RCNN()
       
    

    
    #print('load cfg from file: {}'.format(args.cfg_file_person))
    #cfg_from_file(args.cfg_file_person)
    #assert_and_infer_cfg()
    #maskRCNN_person = Generalized_RCNN()
    if args.visualize:
        save_image = True
    else:
        save_image = False

    if args.cuda:
        maskRCNN_person.cuda()


    if args.load_ckpt_person:
        load_name = args.load_ckpt_person
        print("loading checkpoint for person %s" % (load_name))
        checkpoint_person = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN_person, checkpoint_person['model'])



    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN_car, args.load_detectron)

    maskRCNN_person = mynn.DataParallel(maskRCNN_person, cpu_keywords=['im_info', 'roidb'],minibatch=True, device_ids=[0])  # only support single GPU


    maskRCNN_person.eval()

    print('load cfg from file: {}'.format(args.cfg_file_car))
    cfg_from_file(args.cfg_file_car)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    #cfg.RESNETS.IMAGENET_PRETRAINED = False  # Don't need to load imagenet pretrained weights
    #assert_and_infer_cfg()

    maskRCNN_car = Generalized_RCNN()

    if args.cuda:
        maskRCNN_car.cuda()

    if args.load_ckpt_car:
        load_name = args.load_ckpt_car
        print("loading checkpoint for car %s" % (load_name))
        checkpoint_car = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN_car, checkpoint_car['model'])
        
    maskRCNN_car = mynn.DataParallel(maskRCNN_car, cpu_keywords=['im_info', 'roidb'],minibatch=True, device_ids=[0])  # only support single GPU
    maskRCNN_car.eval()

    if args.image_dir:
        imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    else:
        imglist = args.images
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #imglist.sort(key=lambda f: int(filter(str.isdigit, f)))
    l = imglist
    try:
        imglist = sorted(l,key=lambda x: int(os.path.splitext(x)[0].split('/')[-1]))
    except:
        print('images couldnot be sorted')
    for i in xrange(num_images):
        print('img', i , ' out of ', num_images, ' filename: ', imglist[i].split('/')[-1],' in camera' ,imglist[i].split('/')[-2])
        im = cv2.imread(imglist[i])
        try:
           assert im is not None
        except:
           continue
        timers = defaultdict(Timer)
        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))

        output_name = os.path.basename(im_name) + '.png.txt'
        output_file = os.path.join(args.output_dir, '{}'.format(output_name)) 
        print(output_file)
        text_file = open(output_file, "w")
        cfg_from_file(args.cfg_file_car)
        cls_boxes_car, cls_segms_car, cls_keyps_car,features_car = im_detect_all(maskRCNN_car, im, timers=timers)
        if len(cls_boxes_car[1]) > 0:
           features_car = features_car.data.cpu().numpy()
        
					
        #print(loop,loop2)
				
        #print(distance_matrix)
        #plt.figure()
        #plot_confusion_matrix(distance_matrix, classes=[0,1,2,3,4,5,6],
        #title='Confusion matrix, without normalization')
        #fig = plt.figure()
        #ax = fig.add_subplot(1,1,1)
        #ax.set_aspect('equal')
        #plt.imshow(distance_matrix, interpolation='nearest', cmap=plt.cm.ocean)
        #plt.colorbar()
        #plt.show()
        #fig.savefig('1.png')
        
        count = 0 
        filename = 'finalized_model.txt'
        loaded_model = pickle.load(open(filename, 'rb'))
        for ind,bb in enumerate(cls_boxes_car[1]):
            string = str(count)
            keyps = [k for klist in cls_keyps_car for k in klist]

            bb_new = [bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]]
            features = features_car[ind,:,:,:].flatten()
            pca_feature=[]
            pca_feature.append(np.transpose(features.astype(np.float)))
            #print(pca_feature)
            features = loaded_model.transform(pca_feature)
            features = features[0]#loaded_model.transform(pca_feature)
            if bb[4]<0.3:
                continue
            for bb_ind,val in enumerate(bb_new):
                 string = string + ',' + str(val)
            for kp_ind,kp in enumerate(keyps[ind][0]):
                 string = string +  ',' + str(kp) + ',' + str(keyps[ind][1][kp_ind]) + ',' + str(int(keyps[ind][2][kp_ind]))
            for feature_ind,feature in enumerate(features):
                 string = string +  ',' + str(feature)
            string = string +','+  str(bb[4]) + ',car'
            text_file.write(string) 
            text_file.write('\n') 
            #print(string)
            count = count+1
        cfg_from_file(args.cfg_file_person)
        cls_boxes_person, cls_segms_person, cls_keyps_person,features_person = im_detect_all(maskRCNN_person, im, timers=timers)
        if len(cls_boxes_person[1]) > 0:
            features_person = features_person.data.cpu().numpy()
        for ind,bb in enumerate(cls_boxes_person[1]):
            string = str(count)
            keyps = [k for klist in cls_keyps_person for k in klist]

            bb_new = [bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]]
            features = features_person[ind,:,:,:].flatten()
            pca_feature=[]
            pca_feature.append(np.transpose(features.astype(np.float)))
            #print(pca_feature)
            features = loaded_model.transform(pca_feature)
            features = features[0]#loaded_model.transform(pca_feature)

            #features = loaded_model.transform(np.transpose(features.astype(np.float)))
#            print(features)
            if bb[4]<0.3:
                continue
            for bb_ind,val in enumerate(bb_new):
                 string = string + ',' + str(val)
            for kp_ind,kp in enumerate(keyps[ind][0]):
                 string = string +  ',' + str(kp) + ',' + str(keyps[ind][1][kp_ind]) + ',' + str(int(keyps[ind][2][kp_ind]))
            for feature_ind,feature in enumerate(features):
                 string = string +  ',' + str(feature)
            string = string +','+  str(bb[4]) + ',person'
            text_file.write(string) 
            text_file.write('\n') 
            #print(string)
            count = count+1

        if save_image == True:
         im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
         image_car = vis_utils.vis_one_image_car(
             im[:, :, ::-1],  # BGR -> RGB for visualization
             im_name,
             args.output_dir,
             cls_boxes_car,
             cls_segms_car,
             cls_keyps_car,
             dataset=dataset,
             box_alpha=0.3,
             show_class=True,
             thresh=0.3,
             kp_thresh=0.1
         )
         output_name = os.path.basename(im_name) + '.png' 
         im = cv2.imread(os.path.join(args.output_dir, '{}'.format(output_name)))
         if im is None:
            continue
         continue
         vis_utils.vis_one_image(
             im[:, :, ::-1],  # BGR -> RGB for visualization
             im_name,
             args.output_dir,
             cls_boxes_person,
             cls_segms_person,
             cls_keyps_person,
             dataset=dataset,
             box_alpha=0.3,
             show_class=True,
             thresh=0.3,
             kp_thresh=10
         )

    if args.merge_pdfs and num_images > 1 and save_image==True:
        merge_out_path = '{}/results.pdf'.format(args.output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
