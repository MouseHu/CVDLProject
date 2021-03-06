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
from loader.Synthia_cityscapes import SynthiaCityscapesDataset
from loader.Cityscapes_loader import CityscapesDataset
from loader.Synthia_loader import SynthiaDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from PIL import Image

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
#root_dir="/shuju/segmentation/CityScapes/"
#train_file = os.path.join(root_dir, "train.lst")
#val_file   = os.path.join(root_dir, "val.lst")
#val_data = CityscapesDataset( phase='val', flip_rate=0)

root_dir="/shuju/huhao/synthia-cityscapes/RAND_CITYSCAPES/"
train_file = os.path.join(root_dir, "train.txt")
val_file   = os.path.join(root_dir, "val.txt")
val_data = SynthiaCityscapesDataset(phase='val', flip_rate=0)
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

vgg_model.load_state_dict(torch.load(model_dir+"vgg_model_epoch10.pth").state_dict())
deconv_model.load_state_dict(torch.load(model_dir+"deconv_model_epoch10.pth").state_dict())
criterion = nn.CrossEntropyLoss()




def toImage(pred):
    color={0:[0,0,0],1:[70,130,180],2:[70,70,70],3:[128,64,128],4:[244,35,232],5:[64,64,128],6:[107,142,35],7:[153,153,153],8:[0,0,142],9:[220,220,0],10:[220,20,60],11:[119,11,32],12:[0,0,230],13:[250,170,160],14:[128,64,64],15:[250,170,30],16:[152,251,152],17:[255,0,0],18:[0,0,70],19:[0,60,100],20:[0,80,100],21:[102,102,156],22:[102,102,156]}
    h,w=pred.shape
    #print(pred.shape)
    img=np.zeros((h,w,3))
    #print(img.shape)
    for i in range(h):
	for j in range(w):
		img[i][j]=np.array(color[pred[i][j]])
    return np.uint8(img)

def infer(dataset):
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
	print (vgg_model(inputs)['x5'].shape)
	#output=output[:,:,24:-24,:]
        output = output.data.cpu().numpy()
	#print(2)
        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w).astype(int)
	Image.fromarray(toImage(pred[0])).save("image-"+dataset+"/"+str(iters)+"-ori.jpg")
	print(iters)
	if iters>10:
		break
if __name__ == "__main__":
	infer("syn-cs")

    
    
