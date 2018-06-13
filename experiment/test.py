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
data=open(train_file).readlines()
x=set()
for i in range(0):
	label=scipy.misc.imread('{}/GT/LABELS/{}.png'.format(root_dir, data[i].rstrip("\n")), mode='RGB')
	
	for j in np.unique(label):
		x.add(j)
	if i%100==0:
		print(i,x)
label=scipy.misc.imread('{}/GT/LABELS/{}.png'.format(root_dir, data[2200].rstrip("\n")), mode='YCbCr')
print(label)
print(np.unique(label[:,:,0]),np.unique(label[:,:,2]),np.unique(label[:,:,1]),label.shape)
#print(np.unique(label),label.shape)
