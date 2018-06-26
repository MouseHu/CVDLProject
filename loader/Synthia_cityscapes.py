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


root_dir   = "/shuju/huhao/synthia-cityscapes/RAND_CITYSCAPES/"
train_file = os.path.join(root_dir, "train.txt")
val_file   = os.path.join(root_dir, "test.txt")

num_class = 23
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 760,1280
train_h   = h/2  
train_w   = w/2 
val_h     = h  
val_w     = w  


class SynthiaCityscapesDataset(Dataset):

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
        label_name = self.data[idx].rstrip("\n")
        label      = scipy.misc.imread('{}/GT/COLOR/{}.png'.format(root_dir, self.data[idx].rstrip("\n")), mode='RGB')
	label=self.convert(label)
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


    def convert(self,label):
        labels={(0,0,0):0,(70,130,180):1,(70,70,70):2,(128,64,128):3,(244,35,232):4,(64,64,128):5,(107,142,35):6,(153,153,153):7,(0,0,142):8,(220,220,0):9,(220,20,60):10,(119,11,32):11,(0,0,230):12,(250,170,160):13,(128,64,64):14,(250,170,30):15,(152,251,152):16,(255,0,0):17,(0,0,70):18,(0,60,100):19,(0,80,100):20,(102,102,156):21,(102,102,156):22}

	h,w,_=label.shape
	new_label=np.zeros((h,w))
	#print(new_label.shape)
	for i in range(h):
		for j in range(w):
			try:			
				a=labels[(label[i,j][0],label[i,j][1],label[i,j][2])]
			except KeyError:
				a=-100
			new_label[i,j]=a
	#print(np.unique(new_label))
	return new_label
