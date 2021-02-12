from torchvision import models, transforms
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
import random
		
def OutOfSampleTests(CNNModel, device):
	with torch.no_grad():
		waterBottle = cv2.imread('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\testImages\\waterBottle.jpg', 1)
		orange = cv2.imread('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\testImages\\orange.jpg', 1)
		apple = cv2.imread('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\testImages\\apple.jpg', 1)

		waterBottle = cv2.resize(waterBottle,(128,128))
		orange = cv2.resize(orange, (128,128))
		apple = cv2.resize(apple, (128,128))
		
		waterBottleTensor = torch.Tensor(waterBottle).view(-1,3,128,128)/255
		orangeTensor = torch.Tensor(orange).view(-1,3,128,128)/255
		appleTensor = torch.Tensor(apple).view(-1,3,128,128)/255

		shouldBeWaterBottle = CNNModel(waterBottleTensor.to(device))
		shouldBeOrange = CNNModel(orangeTensor.to(device))
		shouldBeApple =CNNModel(appleTensor.to(device))

		print("my water bottle: ", shouldBeWaterBottle, torch.argmax(shouldBeWaterBottle).item()) # Should be 2, 001 for WB
		print("orange: ", shouldBeOrange, torch.argmax(shouldBeOrange).item()) # Should be 1, 010 for orange
		print("apple: ", shouldBeApple, torch.argmax(shouldBeApple).item()) # Should be 0, 100 for apple


def Testing(testDataX, testDataY, device, CNNModel):
	correct = 0
	correct_wb = 0
	correct_orange = 0
	correct_apple = 0
	total = 0
	with torch.no_grad():
		for i in range(len(testDataX)):
			realClass = torch.argmax(testDataY[i]).to(device)
			netOut = CNNModel(testDataX[i].view(-1,3,128,128).to(device))
			predictedClass = torch.argmax(netOut)
			if predictedClass == realClass:
				correct +=1
				if predictedClass == 2:
					correct_wb +=1
				elif predictedClass == 1:
					correct_orange +=1
				else:
					correct_apple +=1
			total += 1
		return correct/total, correct_wb/(total/3), correct_orange/(total/3), correct_apple/(total/3)

class Model(nn.Module):
	def __init__(self, channel=1, imageSize=128):
		super().__init__()
		# Create a 4 layer convolutional network with batch normalisation
		self.imageSize = imageSize
		self.channel = channel 
		out1 = 32 
		out2 = 64 
		out3 = 128
		out4 = 256
		out5 = 512
		self.conv1 = nn.Conv2d(self.channel, out1, 7, stride=1, padding=2) 
		self.conv2 = nn.Conv2d(out1, out2, 5, stride=1, padding=2)
		self.conv3 = nn.Conv2d(out2, out3, 3, stride=1, padding=2) 
		self.conv4 = nn.Conv2d(out3, out4, 3, stride=1, padding=2)
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
		self.fc1 = nn.Linear(self._to_linear, 1500)
		self.fc2 = nn.Linear(1500, 512)
		self.fc3 = nn.Linear(512, 256)
		self.fc4 = nn.Linear(256, 3)

	def convs(self, x):
		# Convolution --> Batch norm --> Activation func --> Pooling
		x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2,2))
		x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2,2))
		x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2,2))
		x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2,2))
		x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), (2,2))

		if self._to_linear is None: # Determine the number of output neurons of the last convolutional layer. 
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] # This will be the input to the first fully connected layer
			print("Final conv layer output is: ", x.shape)
		return x

	def forward(self,x):
		m = nn.Dropout(0.2)
		x = self.convs(x)
		x = x.view(-1, self._to_linear) # Flatten the convolutional layer output before feeding it in the dense layer
		# Apply the activation function and feed the data in the dense layers
		x = F.relu(self.fc1((x)))
		x = F.relu(self.fc2((x)))
		x = F.relu(self.fc3((x)))
		x = self.fc4(x)
		return x


def Training():
	device = torch.device("cuda:0")
	CNNModel = Model(channel=3).to(device)
	print(CNNModel)
	optimizer = optim.Adam(CNNModel.parameters(), lr=0.004) 
	sched = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	lossFunction = nn.CrossEntropyLoss()

	# Load the training and validation images/labels
	trainDataX = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\Project128files\\TrainingImagesAsTorch.pt')
	trainDataY = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\Project128files\\TrainingLabelsAsTorch.pt')
	testDataX = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\Project128files\\ValidationImagesAsTorch.pt')
	testDataY = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\Project128files\\ValidationLabelsAsTorch.pt')
	trainDataY = trainDataY.type(torch.LongTensor) # For CEL

	batchSize = 64
	epochs = 50
	epoch_loss = np.zeros((epochs+1,1)) # Used to track total loss for each epoch
	epoch_acc = np.zeros((epochs+1,1)) # Used to track accuracy for each epoch
	batch_loss = np.zeros((len(trainDataX)+1,1)) # Used to store the loss for each batch

	transform2 = transforms.Compose([
	transforms.RandomRotation(degrees=175),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	])
# bring linen
# 1150 when pick up the boat
# number of technician: M: nathan brady head technician keiran gave me ur number for extra appliances 0405141390 Site office: 0499880418 lock up compound at mooney mooney. go to office to unload before parking
	for epoch in tqdm(range(epochs)):
		idx = torch.randperm(trainDataX.shape[0])
		trainDataX, trainDataY = trainDataX[idx].view(trainDataX.size()), trainDataY[idx].view(trainDataY.size())
		CNNModel.train()
		for i in range(0, len(trainDataX), batchSize):
			batchX = trainDataX[i:i+batchSize]
			number = random.uniform(0,1)
			if number > 0.5:
				batchX = transform2(batchX)
			batchY = trainDataY[i:i+batchSize]
			batchY = torch.argmax(batchY, dim=1)
			CNNModel.zero_grad()
			output = CNNModel(batchX.to(device))
			loss = lossFunction(output, batchY.to(device))
			loss.backward()
			optimizer.step()
			batch_loss[i,:] = float(loss) # Store the loss for this batch
		# sched.step()
		epoch_loss[epoch+1,:] = np.sum(batch_loss) # Store the total loss for this epoch
		CNNModel.eval()
		epoch_acc[epoch+1,:], wb, orng, appl = Testing(testDataX, testDataY, device, CNNModel) # Store the accuracy for this epoch
		if (epoch_acc[epoch+1,:].item() > epoch_acc[:epoch+1]).all():
			print("\nAt ", epoch+1, "the accuracy is: ", epoch_acc[epoch+1,:])
			torch.save(CNNModel.state_dict(), 'C:\\Users\\louis\\OneDrive\\Desktop\\NN\\5levelCNN\\test.pt')
			print("\nWater bottle accuracy:", round(wb,3), "Orange accuracy:", round(orng,3), "Apple accuracy:", round(appl,3))
			OutOfSampleTests(CNNModel, device)
		if epoch < 10 and epoch_acc[epoch+1,:].item() < 0.34:
			print("Network is not learning, change hyperparameters")
			break

	# Plot the training loss and accuracy
	ceFig, ceAxes = plt.subplots()  
	ceAxes.set_title("Error")  
	ceAxes.grid(True, which='both') 
	ceAxes.plot(np.arange(1,epochs+1),epoch_loss[1:epochs+1,0]) 
	ceAxes.set_xlabel('Epochs')  
	ceAxes.set_ylabel('Error')
	accFig, accAxes = plt.subplots()  
	accAxes.set_title("Accuracy")  
	accAxes.grid(True, which='both') 
	accAxes.plot(np.arange(1,epochs+1),epoch_acc[1:epochs+1,0]) 
	accAxes.set_xlabel('Epochs')  
	accAxes.set_ylabel('Accuracy')
	print("Max accuracy achieved: ", np.max(epoch_acc))
	ind = np.where(epoch_acc == np.max(epoch_acc))
	print("Corresponding loss: ", epoch_loss[ind])
	plt.show()

Training()