from torchvision import models, transforms
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import math 
import time
import cv2
import warnings
import random

class Model(nn.Module):
	def __init__(self, channel=1, imageSize=128):
		super().__init__()
		# Create a 4 layer convolutional network with batch normalisation
		self.imageSize = imageSize
		self.channel = channel 
		out1 = 32 #24 #
		out2 = 64 #28 #
		out3 = 128 #32 #
		out4 = 256 #36 #'
		out5 = 512 # #
		self.conv1 = nn.Conv2d(self.channel, out1, 7, stride=1, padding=2) 
		self.conv2 = nn.Conv2d(out1, out2, 7, stride=1, padding=2)
		self.conv3 = nn.Conv2d(out2, out3, 5, stride=1, padding=2) 
		self.conv4 = nn.Conv2d(out3, out4, 5, stride=1, padding=2)
		self.conv5 = nn.Conv2d(out4, out5, 3, stride=1, padding=2)
		self.bn1 = nn.BatchNorm2d(out1)
		self.bn2 = nn.BatchNorm2d(out2)
		self.bn3 = nn.BatchNorm2d(out3)
		self.bn4 = nn.BatchNorm2d(out4)
		self.bn5 = nn.BatchNorm2d(out5)

		# Use dummy data to find the output shape of the final conv layer
		x = torch.randn(self.channel,self.imageSize,self.imageSize).view(-1,self.channel,self.imageSize,self.imageSize)
		self._to_linear = None
		self.convs(x)

		# 3 Dense layers (input, output and a hidden layer)
		self.fc1 = nn.Linear(self._to_linear, 2000)
		self.fc2 = nn.Linear(2000, 700)
		self.fc3 = nn.Linear(700, 512)
		self.fc4 = nn.Linear(512, 3)

	def convs(self, x):
		# Convolution --> Batch norm --> Activation func --> Pooling
		m = nn.Dropout(0.1)
		x = F.max_pool2d(F.relu(self.bn1(self.conv1((x)))), (2,2))
		x = F.max_pool2d(F.relu(self.bn2(self.conv2((x)))), (2,2))
		x = F.max_pool2d(F.relu(self.bn3(self.conv3((x)))), (2,2))
		x = F.max_pool2d(F.relu(self.bn4(self.conv4((x)))), (2,2))
		x = F.max_pool2d(F.relu(self.bn5(self.conv5((x)))), (2,2))

		if self._to_linear is None: # Determine the number of output neurons of the last convolutional layer. 
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] # This will be the input to the first fully connected layer
			# print("Final conv layer output is: ", x.shape)
		return x

	def forward(self,x):
		m = nn.Dropout(0.5)
		x = self.convs(x)
		x = x.view(-1, self._to_linear) # Flatten the convolutional layer output before feeding it in the dense layer
		# Apply the activation function and feed the data in the dense layers
		x = F.relu(self.fc1((x)))
		x = F.relu(self.fc2((x)))
		x = F.relu(self.fc3((x)))
		x = self.fc4((x))
		return F.softmax(x, dim=1) #x #


def QuestionTwo(img, CNNModel):
	device = torch.device("cuda:0")
	img = cv2.resize(img, (128,128))
	imgTensor = (torch.Tensor(img).to(device))/255
	CNNModel.eval()
	with torch.no_grad():
		netOut = CNNModel(imgTensor.view(-1,3,128,128).to(device))
		print("Probability distribution is:",[round(num, 4) for num in netOut.tolist()[0]])
		predictedClass = torch.argmax(netOut)
		if predictedClass == 2:
			print('water bottle\n')
		elif predictedClass == 1: 
			print('orange\n')
		else:
			print('apple\n')

warnings.filterwarnings("ignore")
CNNModel = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\good\\FinalModel.pt')
cap = cv2.VideoCapture(1)
if(cap.isOpened()==False):
	print("Camera not found\n")

for i in range(3):
	print("Press q to test the network")
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			cv2.imshow('Frame', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	cv2.destroyAllWindows()
	QuestionTwo(frame, CNNModel)
	time.sleep(0.5)
cap.release()