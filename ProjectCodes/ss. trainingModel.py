#!/usr/bin/env python3
from tqdm import tqdm
from convs import Net
from torchvision import models
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

device = torch.device("cuda:0")
net = Net(channel = 3).to(device)
print(net)
optimizer = optim.Adam(net.parameters(), lr=0.001) #optim.SGD(net.parameters(), lr=0.01, momentum=0.5) #
lossFunction = nn.CrossEntropyLoss() #nn.MSELoss() #

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma = 0.1, verbose=True) 

trainDataX = torch.load("C:\\Users\\louis\\OneDrive\\Desktop\\NN\\trainingImagesAsTorch.pt")
trainDataY = torch.load("C:\\Users\\louis\\OneDrive\\Desktop\\NN\\trainingLabelsAsTorch.pt")
testDataX = torch.load("C:\\Users\\louis\\OneDrive\\Desktop\\NN\\validationImagesAsTorch.pt")
testDataY = torch.load("C:\\Users\\louis\\OneDrive\\Desktop\\NN\\validationLabelsAsTorch.pt")
trainDataY= trainDataY.type(torch.LongTensor)# For CEL
batchSize = 64
epochs = 15

# print(type(trainDataY)) # difference between type and dtype
# print(testDataY.type()) # For checking type of tensor

for epoch in tqdm(range(epochs)):
	for i in range(0, len(trainDataX), batchSize):
		batchX = trainDataX[i:i+batchSize]
		batchY = trainDataY[i:i+batchSize] # For MSE. Don't forget to add softmax in convs
		batchY = torch.argmax(batchY, dim=1) #torch.argmax(trainDataY[i:i+batchSize], dim=1) # For CEL
		batchX, batchY = batchX.to(device), batchY.to(device)
		net.zero_grad() 
		output = net(batchX)
		loss = lossFunction(output, batchY)
		loss.backward() # Apply backpropagation
		optimizer.step() # Optimise the parameters
	# scheduler.step()
	print("\nLoss:", round(float(loss),7))

correct = 0
total = 0
with torch.no_grad():
	for i in range(len(testDataX)):
		realClass = torch.argmax(testDataY[i]).to(device)
		netOut = net(testDataX[i].view(-1,3,128,128).to(device))[0]
		predictedClass = torch.argmax(netOut)
		if predictedClass == realClass:
			correct +=1
		total += 1
	print("Accuracy: ", round(correct/total,3))

	torch.save(net, "C:\\Users\\louis\\OneDrive\\Desktop\\NN\\savedModels\\test.pt")

	waterBottle = cv2.imread('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\testImages\\waterBottle.jpg', 1)
	orange = cv2.imread('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\testImages\\orange.jpg', 1)
	apple = cv2.imread('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\testImages\\apple.jpg', 1)

	waterBottle = cv2.resize(waterBottle,(128,128))
	orange = cv2.resize(orange, (128,128))
	apple = cv2.resize(apple, (128,128))
	
	waterBottleTensor = torch.Tensor(waterBottle).view(-1,3,128,128)/255
	orangeTensor = torch.Tensor(orange).view(-1,3,128,128)/255
	appleTensor = torch.Tensor(apple).view(-1,3,128,128)/255

	shouldBeWaterBottle = torch.argmax(net(waterBottleTensor.to(device)))
	shouldBeOrange = torch.argmax(net(orangeTensor.to(device)))
	shouldBeApple = torch.argmax(net(appleTensor.to(device)))

	print("my water bottle: ", shouldBeWaterBottle.item()+1)
	print("orange: ", shouldBeOrange.item()+1)
	print("apple: ", shouldBeApple.item()+1)

	cv2.imshow('waterbottle',waterBottle)
	cv2.imshow('orange',orange)
	cv2.imshow('apple',apple)
	cv2.waitKey(5)
	cv2.destroyAllWindows