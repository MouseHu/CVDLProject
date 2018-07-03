# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from net.deconv import VGGNet,Deconv
from net.distillation import Dis2

from loader.Synthia_loader import SynthiaDataset
from loader.Cityscapes_loader import CityscapesDataset
import numpy as np
import time
import sys
import os
import math

n_class = 13

pad = 24
pad_h = 960
pad_w = 768
source_batch_size = 1
target_batch_size = 1
lambda1 = 0.1   # multiplier of distillation loss
lambda2 = 0.01  # multiplier of spatial-aware adaptation loss
epochs = 500
lr = 1e-4
momentum = 0
w_decay = 1e-51
step_size = 50
gamma = 0.5
configs = "FCNs_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs,
                                                                                                   step_size, gamma, lr,
                                                                                                   momentum, w_decay)
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
model_dir = "road-models/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

#######    load data and models

source_data = SynthiaDataset(phase='train')
source_loader = DataLoader(source_data, batch_size=source_batch_size, shuffle=True, num_workers=8)

target_data = CityscapesDataset(phase='train', flip_rate=0)
target_loader = DataLoader(target_data, batch_size=target_batch_size,shuffle=True, num_workers=8)

val_data = CityscapesDataset(phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

source_vgg_model = VGGNet(model='vgg19', requires_grad=True, remove_fc=True)  #####change to 19
target_vgg_model = VGGNet(model='vgg19', requires_grad=True, remove_fc=True)  # fixed pretrained with ImageNet

deconv_model = Deconv(n_class=n_class)
# domain classifiers
dis_models = []
for i in range(9):
    dis_models.append(Dis2(512))

if use_gpu:
    ts = time.time()
    source_vgg_model = source_vgg_model.cuda()
    target_vgg_model = target_vgg_model.cuda()
    deconv_model = deconv_model.cuda()
    deconv_model = nn.DataParallel(deconv_model, device_ids=num_gpu)
    for dis_model in dis_models:
        dis_model = dis_model.cuda()
        dis_model = nn.DataParallel(dis_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(list(source_vgg_model.parameters())+list(deconv_model.parameters())+
                          list(dis_models[0].parameters())+list(dis_models[1].parameters())+
                          list(dis_models[2].parameters()) + list(dis_models[3].parameters()) +
                          list(dis_models[4].parameters()) + list(dis_models[5].parameters()) +
                          list(dis_models[6].parameters()) + list(dis_models[7].parameters()) +
                          list(dis_models[8].parameters()), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        iter = 0
        for source_batch, target_batch in zip(source_loader,target_loader):
            optimizer.zero_grad()

            if use_gpu:
                source_inputs = Variable(source_batch['X'].cuda())
                source_lefts = Variable(source_batch['left'].cuda())
                source_tops = Variable(source_batch['top'].cuda())
                source_val_ws = Variable(source_batch['val_w'].cuda())
                source_val_hs = Variable(source_batch['val_h'].cuda())
                source_labels = Variable(source_batch['l'].cuda())
                target_inputs = Variable(target_batch['X'].cuda())
                target_lefts = Variable(target_batch['left'].cuda())
                target_tops = Variable(target_batch['top'].cuda())
                target_val_ws = Variable(target_batch['val_w'].cuda())
                target_val_hs = Variable(target_batch['val_h'].cuda())
            else:
                source_inputs = Variable(source_batch['X'])
                source_lefts = Variable(source_batch['left'])
                source_tops = Variable(source_batch['top'])
                source_val_ws = Variable(source_batch['val_w'])
                source_val_hs = Variable(source_batch['val_h'])
                source_labels = Variable(source_batch['l'])
                target_inputs = Variable(target_batch['X'])
                target_lefts = Variable(target_batch['left'])
                target_tops = Variable(target_batch['top'])
                target_val_ws = Variable(target_batch['val_w'])
                target_val_hs = Variable(target_batch['val_h'])
            source_inputs_pad = torch.zeros(source_inputs.shape[0],source_inputs.shape[1],pad_w,pad_h)
            source_inputs_pad[:,:,pad:-pad,:]=source_inputs
            target_inputs_pad = torch.zeros(target_inputs.shape[0],target_inputs.shape[1],pad_w,pad_h)
            target_inputs_pad[:,:,pad:-pad,:]=target_inputs

            if use_gpu:
                source_inputs_pad = source_inputs_pad.cuda()
                target_inputs_pad = target_inputs_pad.cuda()
            source_feature = source_vgg_model(source_inputs_pad)['x5']
            target_seg_feature = source_vgg_model(target_inputs_pad)['x5']
            target_pretrained_feature = target_vgg_model(target_inputs_pad)['x5']
            target_pretrained_feature.detach_() # frozen the pretrained model

            loss_dist = np.square(target_pretrained_feature-target_seg_feature).sum()

            seg_pred = deconv_model(source_feature)
            seg_pred = seg_pred[:,:,pad:-pad,:]

            loss_seg = criterion(seg_pred,source_labels)

            loss_spt = 0
            feature_w = source_feature[0].shape[-1]
            feature_h = source_feature[0].shape[-2]

            for count in range(source_batch_size):
                for i in range(feature_w):
                    for j in range(feature_h):
                        idx = getClassifierIdx(source_val_ws[count],source_val_hs[count],source_tops[count],source_lefts[count],i,j)
                        dis_gt = torch.zeros(1).long()
                        dis_input = source_feature[count,:,i,j]
                        if use_gpu:
                            dis_input = dis_input.cuda()
                        dis_pred = dis_models[idx](dis_input)
                        loss_spt+= criterion(dis_gt,dis_pred)
            for count in range(target_batch_size):
                for i in range(feature_w):
                    for j in range(feature_h):
                        idx = getClassifierIdx(target_val_ws[count],target_val_hs[count],target_tops[count],target_lefts[count],i,j)
                        dis_gt = torch.zeros(1).long()
                        dis_gt[0]=1
                        dis_input = source_feature[count,:,i,j]
                        if use_gpu:
                            dis_input = dis_input.cuda()
                        dis_pred = dis_models[idx](dis_input)
                        loss_spt+= criterion(dis_gt,dis_pred)

            loss_total = loss_seg + lambda1 * loss_dist + lambda2 * loss_spt

            loss_total.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss_total.data[0]))
            iter += 1
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(source_vgg_model, model_dir + "vgg_model_epoch" + str(epoch) + ".pth")
        torch.save(deconv_model, model_dir + "deconv_model_epoch" + str(epoch) + ".pth")
        val(epoch)

def val2(epoch):
    source_vgg_model.eval()
    deconv_model.eval()
    hist = np.zeros((n_class, n_class))
    loss = 0

def val(epoch):
    source_vgg_model.eval()
    deconv_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = source_vgg_model(inputs)
        output = output[:, :, 24:-24, :]
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
    total = (target == target).sum()
    return correct / total

def getClassifierIdx(val_w,val_h,top,left,i,j):
    pos_x=math.floor(left/32)+i
    pos_y=math.floor(top/32)+j
    bound_x1=math.floor(val_w/(32*3))
    bound_x2=math.floor(val_w/(16*3))
    bound_y1 = math.floor(val_h / (32 * 3))
    bound_y2 = math.floor(val_h / (16 * 3))
    if pos_x<=bound_x1:
        if pos_y<=bound_y1:
            return 0
        elif pos_y<=bound_y2:
            return 1
        else:
            return 2
    elif pos_x<=bound_x2:
        if pos_y<=bound_y1:
            return 3
        elif pos_y<=bound_y2:
            return 4
        else:
            return 5
    else:
        if pos_y<=bound_y1:
            return 6
        elif pos_y<=bound_y2:
            return 7
        else:
            return 8


if __name__ == "__main__":
    # val(0)  # show the accuracy before training
    train()
