from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import fcn
class CGAN(nn.Module):
	def __init__(self,in_channels=64,v=512,B=4):
		super(CGAN,self).__init__()
		
		self.conv=nn.Conv2d(in_channels*2, v, kernel_size=3, padding=1)
		self.resblocks=[]
		for i in range(B):
			self.resblocks.append(self.resBlock(v,v,v))
		self.aver_pool=nn.AvgPool2d(kernel_size=16, stride=16)### cun yi
	def resBlock(self,in_channels,v,out_channels):
		layers = []
		layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
		layers += [nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
		layers += [nn.Conv2d(v, out_channels, kernel_size=3, padding=1)]
		result=nn.Sequential(*layers)
		if torch.cuda.is_available():
			result=result.cuda()
		return result

	def forward(self, x):
		noise=torch.randn(x.shape)
		if torch.cuda.is_available():
			noise=noise.cuda()
		out=torch.cat((noise,x),dim=1)
		out=self.conv(out)
		for block in self.resblocks:
			#out2=block(out)
			out=block(out)+out### res link
		out=self.aver_pool(out)
		return out
class Dis(nn.Module):
	def __init__(self,in_channel):
		super(Dis,self).__init__()
		self.fc1=nn.Linear(in_channel,1024)
		self.fc2=nn.Linear(1024,1024)
		self.out=nn.Linear(1024,2)
		self.sigmoid=nn.Sigmoid()
	def forward(self,x):
		r1=self.fc1(x)
		r2=self.fc2(r1)
		out=self.out(r2)
		result=self.sigmoid(out)
		return result
