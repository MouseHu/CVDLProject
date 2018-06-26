# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from net.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs

from loader.Synthia_cityscapes import SynthiaCityscapesDataset
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

# create dir for model
model_dir = "syn-cs-models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)




train_data = SynthiaCityscapesDataset(phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

val_data = SynthiaCityscapesDataset(phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

vgg_model = VGGNet( model='vgg19',requires_grad=True,remove_fc=True)#####change to 19
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)



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
            outputs = fcn_model(inputs2)
	    outputs=outputs[:,:,2:-2,:]
	    #print("here may not be ok")
	    #N, _, h, w = outputs.shape
	
            #pred = torch.max(outputs,dim=1)[1].reshape(N, h, w)
	    #print(outputs.shape,labels.shape)
            loss = criterion(outputs, labels)

	    #labels=labels
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path+str(epoch)+".pth")

        val(epoch)


def val(dataset):
    fcn_model.eval()
    #print("here?"+str(dataset))
    hist = np.zeros((n_class, n_class))
    for iters, batch in enumerate(val_loader):
	#print(0)
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])
	#print(1)
        inputs_pad=torch.zeros(1,3,768,1280)####
	inputs_pad[:,:,4:-4,:]=inputs####
        output = fcn_model(inputs_pad)
        output = output.data.cpu().numpy()
	output=output[:,:,4:-4,:]###
	#print(2)
        N, _, h, w = output.shape
        pred = output.argmax(axis=1).reshape(N, h, w)
	#pred = compat(pred)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
	target[np.where(target==-100)]=0
	#target=compat(target,"val")
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
    val(0)  # show the accuracy before training
    train()
