# -*- coding: utf-8 -*-
###modified by huhao
from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


root_dir   = "/shuju/segmentation/"
train_file = os.path.join(root_dir, "train.lst")
val_file   = os.path.join(root_dir, "val.lst")
label_file   = os.path.join(root_dir, "label.lst")

num_class = 34
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 1024,2048
train2_h   = 380  # 512
train2_w   = 640  # 1024

train_h   = 512  # 512
train_w   = 1024  # 1024

val_h     = h  # 1024
val_w     = w  # 2048
test_h     = 768  # 1024
test_w     = 1280  # 2048


class CityscapesDataset(Dataset):

    def __init__(self, phase, n_class=num_class, crop=False, flip_rate=0.):
	if phase=="train":        
		self.data      = open(train_file).readlines()
		self.label     = open(label_file).readlines()
	else:
		self.data=open(val_file).readlines()
		self.label     = open(label_file).readlines()
        self.means     = means
        self.n_class   = n_class
	self.phase=phase
        self.flip_rate = flip_rate
        self.crop      = crop
        if phase == 'train':
            self.crop = True
            self.flip_rate = 0.5
            self.new_h = train_h
            self.new_w = train_w
	elif phase == 'train2':
            self.crop = True
            self.flip_rate = 0.5
            self.new_h = train2_h
            self.new_w = train2_w
	elif phase == 'test':
	    self.crop = True
            self.flip_rate = 0
            self.new_h = test_h
            self.new_w = test_w
	else:
	    self.new_h = val_h
            self.new_w = val_w
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img        = scipy.misc.imread('{}/cityscapes/{}'.format(root_dir ,self.data[idx].rstrip("\n")), mode='RGB')
        label      = scipy.misc.imread('{}/cityscapes/{}'.format(root_dir, self.label[idx].rstrip("\n")), mode='L')
	#label=self.convert(label)
	#label=self.compatible(label)
        if self.crop:
            h, w, _ = img.shape
            top   = random.randint(0, h - self.new_h)
            left  = random.randint(0, w - self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample


    def convert(self,label):
	labels={(0, 60, 100): 24, (111, 74, 0): 2, (0, 0, 110): 26, (0, 0, 230): 28, (180, 165, 180): 11, (220, 20, 60): 20, (102, 102, 156): 9, (70, 130, 180): 19, (244, 35, 232): 5, (119, 11, 32): 29, (190, 153, 153): 10, (70, 70, 70): 8, (150, 120, 90): 13, (152, 251, 152): 18, (0, 0, 90): 25, (230, 150, 140): 7, (250, 170, 30): 15, (150, 100, 100): 12, (0, 80, 100): 27, (220, 220, 0): 16, (81, 0, 81): 3, (0, 0, 0): 0, (250, 170, 160): 6, (0, 0, 142): 22, (153, 153, 153): 14, (0, 0, 70): 23, (255, 0, 0): 21, (128, 64, 128): 4, (107, 142, 35): 17, (20, 20, 20): 1}

	h,w,_=label.shape
	new_label=np.zeros((h,w))
	#print(new_label.shape)
	for i in range(h):
		for j in range(w):
			try:
				new_label[i,j]=labels[(label[i,j][0],label[i,j][1],label[i,j][2])]
			except KeyError:
				new_label[i,j]=-100
	return new_label
    def compatible(self,label):
	pass
	#convert={1:0,1:1:100,2:100,}
	#for i in range(label.shape[0]):
	#	for j in range(label.shape[1]):
	#		label[i,j]=convert[label[i,j]]
