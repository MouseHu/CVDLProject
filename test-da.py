# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from fcn2 import VGGNet,Deconv
from cgan import CGAN,Dis
from datetime import datetime
from Synthia_cityscapes import SynthiaCityscapesDataset
from Cityscapes_loader import CityscapesDataset
from Synthia_loader import SynthiaDataset
from Cityscapes_loader import CityscapesDataset
from matplotlib import pyplot as plt
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
configs    = "FCNs_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

###
root_dir="/shuju/segmentation/CityScapes/"
train_file = os.path.join(root_dir, "train.lst")
val_file   = os.path.join(root_dir, "val.lst")
val_data = CityscapesDataset( phase='val', flip_rate=0)
val_data2 = CityscapesDataset(phase='train', flip_rate=0)
#val_data = SynthiaCityscapesDataset(phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)
# create dir for model
model_dir = "da-models/"

model_path = os.path.join(model_dir, "test.pth")

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))



vgg_model = VGGNet( model='vgg19',requires_grad=True,remove_fc=True)#####change to 19
deconv_model = Deconv(n_class=n_class)####change to torch.load


if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    vgg_model = nn.DataParallel(vgg_model, device_ids=num_gpu)
    deconv_model = deconv_model.cuda()
    deconv_model = nn.DataParallel(deconv_model, device_ids=num_gpu)
    #gan_model = gan_model.cuda()
    #gan_model = nn.DataParallel(gan_model, device_ids=num_gpu)
    
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

vgg_model.load_state_dict(torch.load(model_dir+"vgg_model_epoch6.pth").state_dict())
deconv_model.load_state_dict(torch.load(model_dir+"deconv_model_epoch6.pth").state_dict())
criterion = nn.CrossEntropyLoss()
#optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)

def compat(target,phase="train"):
###make results comparable
	convert={0:3,1:4,2:13,3:-100,4:10,5:17,6:8,7:18,8:19,9:20,10:12,11:11,12:-100,13:-100
,14:2,15:21,16:5,17:-100,18:-100,19:-100,20:7,21:-100,22:9,23:15,24:6,25:16,26:1,27:0,28:0,29:0,
30:-100,31:-100,32:-100,33:-100,-100:-100}
	new=np.zeros(target.shape).astype(int)
	#print(target.shape)
	for i in range(target.shape[1]):
		for j in range(target.shape[2]):
			new[0,i,j]=int(convert[target[0,i,j]])
	if phase == 'val':	
		new[np.where(new==-100)]=0
	return new
def val(dataset):
    vgg_model.eval()
    deconv_model.eval()
    hist = np.zeros((n_class, n_class))
    loss = 0
    for iters, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])
	#print(1)
        output = deconv_model(vgg_model(inputs)['x5'])
	#output=output[:,:,24:-24,:]
        output = output.data.cpu().numpy()
	#print(2)
        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w).astype(int)
	#pred = compat(pred)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
	target=compat(target,phase="val")
	#print(3)
        hist += fast_hist(pred.flatten(),target.flatten(), n_class)
	print(iters)

    # Calculate average IoU
   
    acc = np.diag(hist).sum() / hist.sum()
    print ('>>>', datetime.now(), 'overall accuracy', acc)
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print ('>>>', datetime.now(),  'mean accuracy', np.nanmean(acc))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('>>>', datetime.now(),'mean IU', np.nanmean(iu))

    print ('>>>', datetime.now(),' IU', iu)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)
# Calculates class intersections over unions
def compute_hist(net, save_dir, dataset,):
    n_cl = net.blobs[layer].channels
    
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)


if __name__ == "__main__":
    val("synthia")  # show the accuracy before training
    #train()
