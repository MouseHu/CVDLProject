from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from net.deconv import VGGNet,Deconv
from net.cgan import CGAN,Dis
from loader.Synthia_cityscapes import SynthiaCityscapesDataset
from loader.Cityscapes_loader import CityscapesDataset
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from itertools import *

#######  super parameter settings
n_class    = 23
alpha=0.5
source_batch_size = 1
target_batch_size = 1
epochs     = 50
lr         = 1e-4
momentum   = 0
w_decay    = 1e-51
step_size  = 50
gamma      = 0.5
configs    = "FCNs_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(source_batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

#######  set model and data dir
source_dir="/shuju/huhao/synthia/"
source_train_file = os.path.join(source_dir, "train.txt")
#val_file   = os.path.join(root_dir, "val.txt")

target_dir="/shuju/segmentation/cityscapes"
target_train_file = os.path.join(target_dir, "train.txt")
target_val_file=os.path.join(target_dir, "val.txt")
# create dir for model
model_dir = "da-models/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

#######    load data and models

source_data = SynthiaCityscapesDataset(phase='train')
source_loader = DataLoader(source_data, batch_size=source_batch_size, shuffle=True, num_workers=8)

target_data = CityscapesDataset(phase='train', flip_rate=0)
target_loader = DataLoader(target_data, batch_size=target_batch_size,shuffle=True, num_workers=8)

val_data = CityscapesDataset(phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)


vgg_model = VGGNet( model='vgg19',requires_grad=True,remove_fc=True)#####change to 19
deconv_model = Deconv(n_class=n_class)####change to torch.load
gan_model= CGAN(B=2)
dis_model=Dis(12*20*512)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    vgg_model = nn.DataParallel(vgg_model, device_ids=num_gpu)
    deconv_model = deconv_model.cuda()
    deconv_model = nn.DataParallel(deconv_model, device_ids=num_gpu)
    gan_model = gan_model.cuda()
    gan_model = nn.DataParallel(gan_model, device_ids=num_gpu)
    dis_model = dis_model.cuda()
    dis_model = nn.DataParallel(dis_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))



criterion = nn.CrossEntropyLoss()

optimizer = optim.RMSprop(list(vgg_model.parameters())+list(gan_model.parameters())+list(dis_model.parameters())+list(deconv_model.parameters()), lr=lr, momentum=momentum, weight_decay=w_decay)

scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
	iters=0
        for  batch1,batch2 in izip(source_loader,target_loader):
            optimizer.zero_grad()
	    #batch2=target_loader[iter]
	    #batch2
            if use_gpu:
                source_inputs = Variable(batch1['X'].cuda())
                source_labels = Variable(batch1['l'].cuda())
		target_inputs = Variable(batch2['X'].cuda())
            else:
                source_inputs, source_labels = Variable(batch1['X']), Variable(batch1['l'])
		target_inputs=Variable(batch2['X'])

	    inputs_pad=torch.zeros(source_inputs.shape[0],source_inputs.shape[1],384,640)
	    inputs_pad[:,:,2:-2,:]=source_inputs
	    target_pad=torch.zeros(target_inputs.shape[0],target_inputs.shape[1],384,640)
	    target_pad[:,:,2:-2,:]=target_inputs
	    #print("inputs_pad",np.prod(list(inputs_pad.shape))*4.0/1024/1024)
	    #print("target_pad",np.prod(list(target_pad.shape))*4.0/1024/1024)
	    if use_gpu:
		inputs_pad=inputs_pad.cuda()
		target_pad=target_pad.cuda()
	    adapt_feature = vgg_model(inputs_pad)['x1']
            source_feature = vgg_model(inputs_pad)['x5']
	    target_feature = vgg_model(target_pad)['x5']
	    #print("adapt_feature",np.prod(list(adapt_feature.shape))*4.0/1024/1024)
	    #print("target_feature",np.prod(list(adapt_feature.shape))*4.0/1024/1024)
	    adapt_feature = gan_model(adapt_feature)+source_feature
	    if iters%2==0:
		source_feature=source_feature.detach()
		target_feature=target_feature.detach()
		adapt_feature=adapt_feature.detach()
	    
	
	    dis_gt=torch.zeros(source_inputs.shape[0]+target_inputs.shape[0]).long()
	    #print(dis_gt)
	    dis_gt[0:source_inputs.shape[0]]=0
	    dis_gt[source_inputs.shape[0]:]=1
	    if use_gpu:
		dis_gt=dis_gt.cuda()
	    dis_input=torch.cat((adapt_feature,target_feature),dim=0)
	    #print("dis_input",np.prod(list(dis_input.shape))*4.0/1024/1024)
	    #print(dis_pred.shape)
	    dis_input=dis_input.view(dis_input.shape[0],-1)
	    dis_pred=dis_model(dis_input)
	    
	    #dis_pred=dis_pred.view(dis_pred.shape[0],-1)
	    dis_loss=criterion(dis_pred,dis_gt)
	    
	    seg_pred1=deconv_model(source_feature)
	    seg_pred1=seg_pred1[:,:,2:-2,:]
	    seg_pred2=deconv_model(adapt_feature)
	    seg_pred2=seg_pred2[:,:,2:-2,:]
	    #print("seg_pred1",np.prod(list(seg_pred1.shape))*4.0/1024/1024)
	    #print("seg_pred2",np.prod(list(seg_pred2.shape))*4.0/1024/1024)
	    if iters%2==1:
		seg_pred1=seg_pred1.detach()
		seg_pred2=seg_pred2.detach()

	    seg_loss1=criterion(seg_pred1, source_labels)
	    seg_loss2=criterion(seg_pred2, source_labels)
	    
	    #N, _, h, w = outputs.shape
	
            #pred = torch.max(outputs,dim=1)[1].reshape(N, h, w)
	    #print(outputs.shape,labels.shape)
            total_loss=(seg_loss1+seg_loss2)/2+alpha*dis_loss
	    #print("total_loss",np.prod(list(total_loss.shape))*4.0/1024/1024)
	    if iters%2==1:###GAN
		total_loss=-total_loss
	    
				
		
            total_loss.backward()
            optimizer.step()

            if iters % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iters, total_loss.item()))
		#print(vgg_model,gan_model,deconv_model.size(),dis_model.size())
	    if iters % 10 == 1:
                print("epoch{}, iter{}, loss: {}".format(epoch, iters, total_loss.item()))
	    iters+=1
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(vgg_model, model_dir+"vgg_model_epoch"+str(epoch)+".pth")
	torch.save(gan_model, model_dir+"gan_model_epoch"+str(epoch)+".pth")
	torch.save(deconv_model, model_dir+"deconv_model_epoch"+str(epoch)+".pth")
	torch.save(dis_model, model_dir+"dis_model_epoch"+str(epoch)+".pth")
        val2(epoch)

def val2(epoch):
    vgg_model.eval()
    gan_model.eval()
    dis_model.eval()
    deconv_model.eval()
    hist = np.zeros((n_class, n_class))
    loss = 0
    
def val(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
	output=output[:,:,24:-24,:]
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
	target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    train()
