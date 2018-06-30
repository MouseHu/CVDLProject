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
from loader.source_target_loader import Source_Target_Dataset
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from itertools import *	
from datetime import datetime



#######  super parameter settings
n_class    = 23
alpha=0.5
source_batch_size = 4
target_batch_size = 4
batch_size = 8
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "batch{}_epoch{}_step{}_gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
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




model_dir = "da-2-models/"
init_model_dir = "syn-cs-models/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

#######    load data and models

source_data = SynthiaCityscapesDataset(phase='train')
source_loader = DataLoader(source_data, batch_size=source_batch_size, shuffle=True, num_workers=8)

target_data = CityscapesDataset(phase='train2', flip_rate=0)
target_loader = DataLoader(target_data, batch_size=target_batch_size,shuffle=True, num_workers=8)

val_data = CityscapesDataset(phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

source_target_data=Source_Target_Dataset(phase='train')
source_target_loader=DataLoader(source_target_data, batch_size=2, num_workers=8)

vgg_model = VGGNet( model='vgg19',requires_grad=True,remove_fc=True)#####change to 19
deconv_model = Deconv(n_class=n_class)####change to torch.load
gan_model= CGAN(B=2)
dis_model=Dis(12*20*512)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    #vgg_model = nn.DataParallel(vgg_model, device_ids=num_gpu)
    deconv_model = deconv_model.cuda()
    deconv_model = nn.DataParallel(deconv_model, device_ids=num_gpu)
    gan_model = gan_model.cuda()
    gan_model = nn.DataParallel(gan_model, device_ids=num_gpu)
    dis_model = dis_model.cuda()
    dis_model = nn.DataParallel(dis_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))


#criterion1 = GANLoss()
criterion2 = nn.CrossEntropyLoss()
criterion3 = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(list(vgg_model.parameters())+list(gan_model.parameters())+list(dis_model.parameters())+list(deconv_model.parameters()), lr=lr, momentum=momentum, weight_decay=w_decay)
optimizerG = optim.RMSprop(list(vgg_model.parameters())+list(gan_model.parameters()), lr=lr, momentum=momentum, weight_decay=w_decay)
optimizerD = optim.RMSprop(list(dis_model.parameters())+list(deconv_model.parameters()), lr=lr*0.1, momentum=momentum, weight_decay=w_decay)

scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

class_dict={0:"void",1:"sky",2:"Building",3:"Road",4:"Sidewalk",5:"Fence",6:"Vegetation",7:"Pole",8:"Car",9:"Traffic sign",10:"Pedestrian",11:"Bicycle",12:"Motorcycle",13:"Parking-slot",14:"Road-work",15:"Traffic light",16:"Terrain",17:"Rider",18:"Truck",19:"Bus",20:"Train",21:"Wall",22:"Lanemarking"}



###continue training
last_epoch=0
init_load=False
init_epoch=17
load=False
if init_load==True:
	vgg_model.load_state_dict(torch.load(init_model_dir+configs+"vgg{}.pth".format(init_epoch)).state_dict())	
	deconv_model.load_state_dict(torch.load(init_model_dir+configs+"deconv{}.pth".format(init_epoch)).state_dict())
if load==True:
	vgg_model.load_state_dict(torch.load(model_dir+"vgg_model_epoch{}.pth".format(last_epoch)).state_dict())
	deconv_model.load_state_dict(torch.load(model_dir+"deconv_model_epoch{}.pth".format(last_epoch)).state_dict())
	gan_model.load_state_dict(torch.load(model_dir+"gan_model_epoch{}.pth".format(last_epoch)).state_dict())
	dis_model.load_state_dict(torch.load(model_dir+"dis_model_epoch{}.pth".format(last_epoch)).state_dict())

def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
	iters=0
        for batch1,batch2 in source_target_loader:
	    #print(batch1,batch2)
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
	 #   print(source_inputs.shape,target_inputs.shape)
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
#	    if iters%2==0:
#		source_feature=source_feature.detach()
#		target_feature=target_feature.detach()
#		adapt_feature=adapt_feature.detach()
	    
	
	    #dis_gt=torch.zeros(source_inputs.shape[0]+target_inputs.shape[0]).long()
	    #print(dis_gt)
	    #dis_gt[0:source_inputs.shape[0]]=0
	    ##dis_gt[source_inputs.shape[0]:]=1
	    #if use_gpu:
		#dis_gt=dis_gt.cuda()
	    
	    dis_input=torch.cat((adapt_feature,target_feature),dim=0)
	    #print("dis_input",np.prod(list(dis_input.shape))*4.0/1024/1024)
	    #print(adapt_feature.shape,target_feature.shape,dis_input.shape)
	    dis_input=dis_input.view(dis_input.shape[0],-1)
	    dis_pred=dis_model(dis_input)
	    #print(dis_input.shape,dis_pred.shape)
	    pred=torch.Tensor(dis_pred.shape)
	    pred[0:source_inputs.shape[0]]=1-dis_pred[0:source_inputs.shape[0]]
	    pred[source_inputs.shape[0]:]=dis_pred[source_inputs.shape[0]:]
	    #print(pred)
	    #dis_pred=dis_pred.view(dis_pred.shape[0],-1)
	    dis_loss=-torch.sum(torch.log(dis_pred))
	    
	    seg_pred1=deconv_model(source_feature)
	    seg_pred1=seg_pred1[:,:,2:-2,:]

	    seg_pred2=deconv_model(adapt_feature)
	    seg_pred2=seg_pred2[:,:,2:-2,:]
	    #print("seg_pred1",np.prod(list(seg_pred1.shape))*4.0/1024/1024)
	    #print("seg_pred2",np.prod(list(seg_pred2.shape))*4.0/1024/1024)
#	    if iters%2==1:
#		seg_pred1=seg_pred1.detach()
#		seg_pred2=seg_pred2.detach()

	    seg_loss1=criterion2(seg_pred1, source_labels)
	    seg_loss2=criterion3(seg_pred2, source_labels)
	    
	    #N, _, h, w = outputs.shape
	
            #pred = torch.max(outputs,dim=1)[1].reshape(N, h, w)
	    #print(outputs.shape,labels.shape)
            if iters%2==0:
	    	total_loss=(seg_loss1+seg_loss2)/2-alpha*dis_loss
		total_loss.backward()
		optimizerG.step()
	    else:
		total_loss=(seg_loss1+seg_loss2)/2+alpha*dis_loss
		total_loss.backward()
		optimizerD.step()
	    #print("total_loss",np.prod(list(total_loss.shape))*4.0/1024/1024)
#	    if iters%2==1:###GAN
#		total_loss=-total_loss
	    
				
		
            
            

            if iters % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch+last_epoch, iters, total_loss.item()))
		print("seg_loss1:{},seg_loss2:{},dis_loss:{}".format(seg_loss1,seg_loss2,dis_loss))
		#print(vgg_model,gan_model,deconv_model.size(),dis_model.size())
	    if iters % 10 == 1:
                print("epoch{}, iter{}, loss: {}".format(epoch+last_epoch, iters, total_loss.item()))
		print("seg_loss1:{},seg_loss2:{},dis_loss:{}".format(seg_loss1,seg_loss2,dis_loss))
	    iters+=1
        
        print("Finish epoch {}, time elapsed {}".format(epoch+last_epoch, time.time() - ts))
        torch.save(vgg_model, model_dir+"3vgg_model_epoch"+str(epoch+last_epoch)+".pth")
	torch.save(gan_model, model_dir+"3gan_model_epoch"+str(epoch+last_epoch)+".pth")
	torch.save(deconv_model, model_dir+"3deconv_model_epoch"+str(epoch+last_epoch)+".pth")
	torch.save(dis_model, model_dir+"3dis_model_epoch"+str(epoch+last_epoch)+".pth")
        val(epoch)

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
        acc = np.diag(hist).sum() / hist.sum()
    print ('>>>', datetime.now(), 'overall accuracy', acc)
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print ('>>>', datetime.now(),  'mean accuracy', np.nanmean(acc))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('>>>', datetime.now(),'mean IU', np.nanmean(iu))

    print ('>>>', datetime.now(),' IU', iu)
    for i in range(n_class):
	print(class_dict[i],iu[i])



def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)
# Calculates class intersections over unions


if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    train()
