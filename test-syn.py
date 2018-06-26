# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from net.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs

from loader.Synthia_loader import SynthiaDataset
from loader.Cityscapes_loader import CityscapesDataset
from loader.Synthia_cityscapes import SynthiaCityscapesDataset
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import time
import sys
import os


n_class    = 23

batch_size = 5
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5


###
root_dir="/shuju/segmentation/CityScapes/"
train_file = os.path.join(root_dir, "train.lst")
val_file   = os.path.join(root_dir, "val.lst")
val_data = CityscapesDataset( phase='val', flip_rate=0)
#val_data2 = CityscapesDataset(phase='train', flip_rate=0)
#val_data = SynthiaCityscapesDataset(phase='val', flip_rate=0)


#root_dir="/shuju/huhao/synthia-cityscapes/RAND_CITYSCAPES/"
#train_file = os.path.join(root_dir, "train.txt")
#val_file   = os.path.join(root_dir, "val.txt")
######here
#val_data = SynthiaCityscapesDataset(phase='train', flip_rate=0)

class_dict={0:"void",1:"sky",2:"Building",3:"Road",4:"Sidewalk",5:"Fence",6:"Vegetation",7:"Pole",8:"Car",9:"Traffic sign",10:"Pedestrian",11:"Bicycle",12:"Motorcycle",13:"Parking-slot",14:"Road-work",15:"Traffic light",16:"Terrain",17:"Rider",18:"Truck",19:"Bus",20:"Train",21:"Wall",22:"Lanemarking"}

val_loader = DataLoader(val_data, batch_size=1, num_workers=8)
# create dir for model


model_dir = "syn-cs-models/"
model_path = os.path.join(model_dir, "test.pth")
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))



vgg_model = VGGNet( model='vgg19',requires_grad=True,remove_fc=True)#####change to 19
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
fcn_model.load_state_dict(torch.load(model_dir+"batch8_epoch500_step50_gamma0.5_lr0.0001_momentum0_w_decay1e-0523.pth").state_dict())

criterion = nn.CrossEntropyLoss()
#optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs



def compat(target,phase="train"):
###make results comparable
	#convert={0:3,1:4,2:13,3:-100,4:10,5:17,6:8,7:18,8:19,9:20,10:12,11:11,12:-100,13:-100
#,14:2,15:21,16:5,17:-100,18:-100,19:-100,20:7,21:-100,22:9,23:15,24:6,25:16,26:1,27:0,28:0,29:0,
#30:-100,31:-100,32:-100,33:-100,-100:-100}
	convert={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:3,8:4,9:13,10:0,11:2,12:21,13:5,14:0,15:0,16:0,17:7,18:0,19:15,20:9,21:6,22:0,23:1,24:10,25:17
,26:8,27:18,28:19,29:0,30:0,31:20,32:12,33:11,-100:-100}
	new=np.zeros(target.shape).astype(int)
	#print(target.shape)
	for i in range(target.shape[1]):
		for j in range(target.shape[2]):
			new[0,i,j]=int(convert[target[0,i,j]])
	if phase == 'val':	
		new[np.where(new==-100)]=0
	return new
def val(dataset):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    hist = np.zeros((n_class, n_class))
    for iters, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])
	if dataset == 'synthia':
        	inputs_pad=torch.zeros(1,3,384,640)####
		inputs_pad[:,:,2:-2,:]=inputs####
        	output = fcn_model(inputs_pad)
        	output = output.data.cpu().numpy()
		output=output[:,:,2:-2,:]###
	else:
		output = fcn_model(inputs)
        	output = output.data.cpu().numpy()
        N, _, h, w = output.shape
        pred = output.argmax(axis=1).reshape(N, h, w)
	#pred = compat(pred)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
	target[np.where(target==-100)]=0
	target=compat(target,"val")
	hist += fast_hist(pred.flatten(),target.flatten(), n_class)
        print(iters)
	if iters%10==0:
		print(np.unique(pred),np.unique(target))
    # Calculate average IoU
    acc = np.diag(hist).sum() / hist.sum()
    print ('>>>', datetime.now(), 'overall accuracy', acc)
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print ('>>>', datetime.now(),  'mean accuracy', np.nanmean(acc))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('>>>', datetime.now(),'mean IU', np.nanmean(iu))
    print ('>>>', datetime.now(),'IU', iu)
    for i in range(n_class):
	print(class_dict[i],iu[i])

def fast_hist(a, b, n):
    
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)



if __name__ == "__main__":
    val("cs")  # show the accuracy before training
    #train()
