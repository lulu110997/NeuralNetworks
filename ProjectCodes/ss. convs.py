#!/usr/bin/env python3
# Followed tutorial from sentdex
import torch
import torch.nn as nn
import torch.nn.functional as F
# 227207723 parameters will not fit in a 6GB GPU
class Net(nn.Module):
	def __init__(self, channel=1, imageSize=128):
		super().__init__()
		# Conv2d is used on images since they are 2D while Conv3d is used in scans or models. 1D is used on sequential or temporal dataset
		# Make a convolutional layer with n input channels, m convolution feature output and a axb kernel
		self.imageSize = imageSize
		self.channel = channel 
		self.conv1 = nn.Conv2d(self.channel, 64, 7) 
		self.conv2 = nn.Conv2d(64, 128, 3)
		self.conv3 = nn.Conv2d(128, 128, 3) 
		# self.conv4 = nn.Conv2d(128, 512, 5)
		# self.conv5 = nn.Conv2d(256, 425, 5)
		# self.conv6 = nn.Conv2d(256, 512, 5)

		# Output shape can be calculated by O = { (W - k + 2*P)/s } + 1
		# Alternatively, pass in a dummy data as below. Run forward pass to find self._to_linear
		x = torch.randn(self.channel,self.imageSize,self.imageSize).view(-1,self.channel,self.imageSize,self.imageSize)
		self._to_linear = None
		self.convs(x)

		self.fc1 = nn.Linear(self._to_linear, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, 3)
		# self.fc4 = nn.Linear(512, 3)

	def convs(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (3,3))
		x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
		# x = F.max_pool2d(F.relu(self.conv4(x)), (2,2), stride=1)
		# x = F.relu(self.conv5(x))
		# x = F.max_pool2d(F.relu(self.conv6(x)), (2,2), stride=1)

		# print(x[0].shape)
		if self._to_linear is None: # Determine the number of output neurons of the last convolutional layer. 
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] # This will be the input to the first fully connected layer
			print("Final conv layer output is: ", x.shape)
		return x

	def forward(self,x):
		x = self.convs(x)
		x = x.view(-1, self._to_linear) # Flatten the convolutional layer output before feeding it in the dense layer
		# Apply the activation function and feed the data in the dense layers
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x)) 
		# x = F.relu(self.fc3(x)) 
		x = self.fc3(x)
		# return F.softmax(x, dim=1)
		return x



def main():
	net = Net()
	print(net)
	total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	total_params_trainable = sum(p.numel() for p in net.parameters())
	print("total parameters: ", total_params)
	print("total trainable parameters: ", total_params_trainable)

if __name__ == "__main__":
	main()