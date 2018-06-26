# -*- coding: utf-8 -*-

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


root_dir   = "/shuju/huhao/synthia/"
train_file = os.path.join(root_dir, "train.txt")
val_file   = os.path.join(root_dir, "test.txt")

num_class = 13
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 720,960 
train_h   = h  
train_w   = w  
val_h     = h  
val_w     = w  


class SynthiaDataset(Dataset):

    def __init__(self, phase, n_class=num_class, crop=False, flip_rate=0.):
        if phase == 'train':
		self.data=open(train_file).readlines()
	else:
		self.data=open(val_file).readlines()
        self.means     = means
        self.n_class   = n_class

        self.flip_rate = flip_rate
        self.crop      = crop
        if phase == 'train':
            self.crop = True
            self.flip_rate = 0.5
            self.new_h = train_h
            self.new_w = train_w

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img        = scipy.misc.imread('{}/RGB/{}.png'.format(root_dir, self.data[idx].rstrip("\n")), mode='RGB')
        #label_name = self.data.ix[idx, 1]
        label      = self.load_label(self.data[idx].rstrip("\n"))

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
	target=target.long()
        sample = {'X': img, 'Y':target ,'l': label}

        return sample


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = open('{}/GTTXT/{}.txt'.format(root_dir, idx))
	#print(type(im.readlines()[0].rstrip("\n")))
        rgb_label = [i.rstrip("\n").split(" ") for i in im.readlines()]
	label=[]	
	for i in rgb_label:
		label+=[int(j) for j in i]
	label=np.array(label).reshape(720,960)
	label[label==-1]=12
	#print(np.unique(label))
        #label = label[np.newaxis, ...]
        return label
