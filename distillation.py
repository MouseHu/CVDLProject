from __future__ import print_function

import torch
import torch.nn as nn

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class Dis2(nn.Module):
	def __init__(self,in_channel):
		super(Dis2,self).__init__()
		self.fc1=nn.Linear(in_channel,1024)
		self.fc2=nn.Linear(1024,1024)
		self.out=nn.Linear(1024,2)
		self.sigmoid=nn.Sigmoid()
	def forward(self, x):
		x = grad_reverse(x)
		r1 = self.fc1(x)
		r2 = self.fc2(r1)
		out = self.out(r2)
		result = self.sigmoid(out)
		return result





