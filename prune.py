import torch
import torch.nn as nn
import numpy as np
import FLOP

PRUNE_ID = 0
DEBUG_MODE = False

## This code prunes all the Conv2d layers in a given pytorch model. The Conv2d are pruned by removing
## channels based on an evaluation of their weights. The pruning is done with these restrictions:
## 1. Each Conv2d after pruning will retain at least 1 channel
## 2. Conv2d layers with groups != 1 or bias != False are not pruned
## After pruning, a zero_padding layer is added to pad the output tensor up to the correct dimensions

## To use the pruning, write something like model = prune_model(model, factor_removed=)
## args: model - your pytorch model
##       factor_removed - the proportion of layers pruning will try to remove

## Idea is from 'Pruning Filters for Efficient ConvNets' by Hao Li, et al
## (https://arvix.org/abs/1608.08710)

class debug_mode(object):
    def __enter__(self):
    	global DEBUG_MODE
    	self.prev = DEBUG_MODE
    	DEBUG_MODE = True
    def __exit__(self, *args):
        DEBUG_MODE = self.prev

def mask(model, cut_off=0):
	for p in model.parameters():
		p_mask = abs(p)>cut_off
		p *= p_mask.float()
	return model

def layer_eval(layer):
	element_squared = [e.item()**2 for e in layer.view(-1)]
	return sum(element_squared)

def unwrap_model(model):
	# loops through all layers of model, including inside modulelists and sequentials
	layers = []
	def unwrap_inside(modules):
		for m in modules.children():
			if isinstance(m,nn.Sequential):
				unwrap_inside(m)
			elif isinstance(m,nn.ModuleList):
				for m2 in m:
					unwrap_inside(m2)
			else:
				layers.append(m)
	unwrap_inside(model)
	return nn.ModuleList(layers)

class zero_padding(nn.Module):
	#my version of zero padding, pads up to givenn number of channels, at the specified index
	def __init__(self, num_channels, keep_channel_idx):
		super(zero_padding, self).__init__()
		self.num_channels = num_channels
		self.keep_channel_idx = keep_channel_idx
	def forward(self,x):
		output = torch.zeros(x.size()[0],self.num_channels,x.size()[2],x.size()[3])
		output[:,self.keep_channel_idx,:,:] = x
		return output

class pruned_conv2d(nn.Module):
	def __init__(self, conv2d, cut_off=0.0):
		super(pruned_conv2d, self).__init__()
		self.in_channels = conv2d.in_channels
		self.out_channels = conv2d.out_channels
		self.kernel_size = conv2d.kernel_size
		self.stride = conv2d.stride
		self.padding = conv2d.padding
		self.dilation = conv2d.dilation
		self.groups = conv2d.groups
		self.bias = conv2d.bias
		global PRUNE_ID
		self.id = PRUNE_ID
		PRUNE_ID+=1
		self.keep_channel = []
		self.keep_channel_idx = []

		if self.groups != 1 or self.bias != None:
			self.new_conv2d = conv2d
		else:
			for idx, channel in enumerate(conv2d.weight):
				if layer_eval(channel)>cut_off:
					self.keep_channel.append(torch.unsqueeze(channel,0))
					self.keep_channel_idx.append(idx)
			if len(self.keep_channel_idx) == 0:
				# if no channels are above cut-off, keep the best channel
				best_channel_eval = 0
				for idx, channel in enumerate(conv2d.weight):
					if layer_eval(channel) > best_channel_eval:
						best_channel = channel
						best_channel_idx = idx
				self.keep_channel.append(torch.unsqueeze(best_channel,0))
				self.keep_channel_idx.append(best_channel_idx)
			self.new_conv2d = nn.Conv2d(in_channels=self.in_channels,
										out_channels=len(self.keep_channel_idx),
										kernel_size=self.kernel_size,
										stride=self.stride,
										padding=self.padding,
										dilation=self.dilation,
										bias=False)
			self.new_conv2d.weight = torch.nn.Parameter(torch.cat(self.keep_channel,0))
			self.zero_padding = zero_padding(self.out_channels, self.keep_channel_idx)

	def forward(self,x):
		if self.groups != 1 or self.bias != None:
			return self.new_conv2d(x)
		else:
			if DEBUG_MODE:
				try:
					x = self.new_conv2d(x)
				except Exception as e:
					print('failed here')
					print('input size: '+ str(x.size()))
					print('layer: ' + str(self.new_conv2d))
					print('layer weight: ' +str(self.new_conv2d.weight.size()))
					print(str(e))
					quit()
			else:
				x = self.new_conv2d(x)
			return self.zero_padding(x)

class prune_model(nn.Module):
	def __init__(self, model, factor_removed=0.75):
		super(prune_model,self).__init__()
		self.model = model
		self.factor = factor_removed
		self.modulelist = unwrap_model(self.model)

		print('number of parameters before pruning: %d' %sum([p.numel() for p in self.model.parameters()]))
		print('FLOP: %d' %FLOP.count_model_param_flops(self.model, 300))
		self.layer_eval_list = []
		for m in self.modulelist:
			if m.__class__.__name__ == "Conv2d":
				for layer in m.weight:
					self.layer_eval_list.append(layer_eval(layer))

		self.layer_eval_list.sort()
		self.cut_off = self.layer_eval_list[int(factor_removed*len(self.layer_eval_list))]

		def replace_inside(modules):
			for name, m in iter(modules._modules.items()):
				if isinstance(m,nn.Sequential):
					replace_inside(m)
				elif isinstance(m,nn.ModuleList):
					for m2 in m:
						replace_inside(m2)
				else:
					if m.__class__.__name__ == "Conv2d":
						modules._modules[name] = pruned_conv2d(m, self.cut_off)
		replace_inside(self.model)
		
		print('number of parameters after pruning: %d' %sum([p.numel() for p in self.model.parameters()]))
		print('FLOP: %d' %FLOP.count_model_param_flops(self.model, 300))
	def forward(self,x, phase='eval', use_RNN=False):
		return self.model(x, phase, use_RNN)



