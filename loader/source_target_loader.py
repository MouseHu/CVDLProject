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

from loader.Cityscapes_loader import CityscapesDataset
from loader.Synthia_cityscapes import SynthiaCityscapesDataset

root_dir   = "/shuju/huhao/synthia/"
train_file = os.path.join(root_dir, "train.txt")
val_file   = os.path.join(root_dir, "test.txt")

root_dir   = "/shuju/segmentation/"
train2_file = os.path.join(root_dir, "train.lst")
val2_file   = os.path.join(root_dir, "val.lst")
label_file   = os.path.join(root_dir, "label.lst")

class Source_Target_Dataset(Dataset):
    def __init__(self, phase, n_class1=23,n_class2=34):
	self.source_dataset=SynthiaCityscapesDataset(phase='train', n_class=n_class1, crop=False, flip_rate=0.)
        self.target_dataset=CityscapesDataset(phase='train2', n_class=n_class2, crop=False, flip_rate=0.)
	
	self.s_l=len(self.source_dataset.data)
	self.t_l=len(self.target_dataset.data)
	self.length=max(len(self.source_dataset.data),len(self.target_dataset.data))
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
	return (self.source_dataset.__getitem__(idx%self.s_l),self.target_dataset.__getitem__(idx%self.t_l))
