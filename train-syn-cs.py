# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from net.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from net.fcn2 import VGGNet,Deconv
from loader.Synthia_cityscapes import SynthiaCityscapesDataset
from loader.Cityscapes_loader import CityscapesDataset
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from datetime import datetime

n_class    = 23

batch_size = 8
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "batch{}_epoch{}_step{}_gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
###
root_dir="/shuju/huhao/synthia-cityscapes/"
train_file = os.path.join(root_dir, "train.txt")
val_file   = os.path.join(root_dir, "val.txt")

root_dir2="/shuju/segmentation/CityScapes/"
#train_file = os.path.join(root_dir, "train.lst")
val2_file   = os.path.join(root_dir2, "val.lst")
# create dir for model
model_dir = "syn-cs-models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)




train_data = SynthiaCityscapesDataset(phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

val_data = SynthiaCityscapesDataset(phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

val2_data = CityscapesDataset(phase='val', flip_rate=0)
val2_loader = DataLoader(val2_data, batch_size=1, num_workers=8)

vgg_model = VGGNet( model='vgg19',requires_grad=True,remove_fc=True)#####change to 19
deconv_model = Deconv( n_class=n_class)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    deconv_model = deconv_model.cuda()
    deconv_model = nn.DataParallel(deconv_model, device_ids=num_gpu)
    #vgg_model = nn.DataParallel(vgg_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(list(vgg_model.parameters())+list(deconv_model.parameters()), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

last_epoch=0
#vgg_model.load_state_dict(torch.load(model_path+"vgg"+str(last_epoch)+".pth").state_dict())
#deconv_model.load_state_dict(torch.load(model_path+"deconv"+str(last_epoch)+".pth").state_dict())



def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
	print("here is ok")
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['l'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['l'])

	    inputs2=torch.zeros(inputs.shape[0],inputs.shape[1],384,640)
	    inputs2[:,:,2:-2,:]=inputs
	    #print("ok2")
	    if use_gpu:
		inputs2=inputs2.cuda()

            outputs = deconv_model(vgg_model(inputs2)['x5'])
	    outputs=outputs[:,:,2:-2,:]
	    
            loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))
        
        print("Finish epoch {}, time elapsed {}".format(last_epoch+epoch, time.time() - ts))
	torch.save(vgg_model, model_path+"vgg"+str(last_epoch+epoch)+".pth")
        torch.save(deconv_model, model_path+"deconv"+str(epoch+last_epoch)+".pth")
        #val(epoch+last_epoch)
	val(epoch+last_epoch,2)
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

def val(dataset,val_type=1):
    vgg_model.eval()
    deconv_model.eval()
    #print("here?"+str(dataset))
    hist = np.zeros((n_class, n_class))
    if val_type==1:
	myval_loader=val_loader
    else:
	myval_loader=val2_loader
    for iters, batch in enumerate(myval_loader):
	#print(0)
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])
	#print(1)
	if val_type==1:
        	inputs_pad=torch.zeros(1,3,768,1280)####
		inputs_pad[:,:,4:-4,:]=inputs####
		if use_gpu:
			inputs_pad=inputs_pad.cuda()
	else:
		inputs_pad=inputs
        output = deconv_model(vgg_model(inputs_pad)['x5'])
        output = output.data.cpu().numpy()
	if val_type==1:	
		output=output[:,:,4:-4,:]###
	#print(2)
        N, _, h, w = output.shape
        pred = output.argmax(axis=1).reshape(N, h, w)
	if val_type==2:	
		pred = compat(pred)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
	target[np.where(target==-100)]=0
	#target=compat(target,"val")
	#print(np.unique(pred),np.unique(target))
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
    print ('>>>', datetime.now(),'IU', iu)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


if __name__ == "__main__":
    val(0,2)  # show the accuracy before training
    train()
