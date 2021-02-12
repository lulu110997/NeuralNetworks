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
import random

def Testing(testDataX, testDataY, device, CNNModel):
	correct = 0
	total = 0
	with torch.no_grad():
		for i in range(len(testDataX)):
			realClass = torch.argmax(testDataY[i]).to(device)
			netOut = CNNModel(testDataX[i].view(-1,3,224,224).to(device))
			predictedClass = torch.argmax(netOut)
			if predictedClass == realClass:
				correct +=1
			total += 1
		return correct/total

device = torch.device("cuda:0")
transform2 = transforms.Compose([
	transforms.RandomRotation(degrees=(135)),
	transforms.ColorJitter(),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	])
transform = transforms.Compose([
	transforms.Normalize(
	mean=[0.485, 0.456, 0.406],
	std=[0.229, 0.224, 0.225]
	)])
# trainDataX = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD256TrainingImages.pt')
# trainDataY = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD256TrainingLabels.pt')
# testDataX = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD256TestingImages.pt')
# testDataY = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD256TestingLabels.pt')
trainDataX = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\AlexTrainingImages.pt')
trainDataY = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\AlexTrainingLabels.pt')
testDataX = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\AlexTestingImages.pt')
testDataY = torch.load('C:\\Users\\louis\\OneDrive\\Desktop\\NN\\AlexTestingLabels.pt')
trainDataY= trainDataY.type(torch.LongTensor)
trainDataX = transform(trainDataX)
testDataX = transform(testDataX)
alexnet = models.resnet18(pretrained=True).to(device)
alexnet = nn.Sequential(models.resnet18(pretrained=True), nn.Dropout(0.4), nn.Linear(128, 3)).to(device)
print(alexnet)
c = 0
for name, param in alexnet.named_parameters():
	if c == 30:
		break
	param.requires_grad = False
	c+=1
	# print(c, name)

alexnet[0].fc = nn.Linear(512, 128).to(device)

params_to_update = alexnet.parameters()
print("Params to learn:")
for name,param in alexnet.named_parameters():
	if param.requires_grad == True:
		print("\t",name)

optimiser = optim.Adam(alexnet.parameters(), lr=0.002) # turn to 0.004
exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.5) # half the step size and gamma
loss_function = nn.CrossEntropyLoss()
batch_size = 128
epochs = 100
batch_loss = np.zeros((len(trainDataX), 1))
epoch_loss = np.zeros((epochs+1, 1))
epoch_acc = np.zeros((epochs+1, 1))

for epoch in tqdm(range(epochs)):
	idx = torch.randperm(trainDataX.shape[0])
	trainDataX, trainDataY = trainDataX[idx].view(trainDataX.size()), trainDataY[idx].view(trainDataY.size())
	alexnet.train()
	for i in range(0, len(trainDataX), batch_size):
		batch_x = trainDataX[i:batch_size+i].to(device)
		number = random.uniform(0, 1)
		if number >= 0.8:
			batch_x = transform2(batch_x)
		batch_y = torch.argmax(trainDataY[i:batch_size+i].to(device),dim=1)
		alexnet.zero_grad()
		output = alexnet(batch_x)
		loss = loss_function(output, batch_y)
		loss.backward()
		optimiser.step()
		batch_loss[i] = float(loss)
	exp_lr_scheduler.step()
	epoch_loss[epoch+1,:] = np.sum(batch_loss) # Store the total loss for this epoch
	alexnet.eval()
	epoch_acc[epoch+1,:] = Testing(testDataX, testDataY, device, alexnet) # Store the accuracy for this epoch
	if (epoch_acc[epoch+1,:].item() > epoch_acc[:epoch+1]).all():
		print("\nAt ", epoch+1, "the accuracy is: ", epoch_acc[epoch+1,:])
		torch.save(alexnet.state_dict(), 'C:\\Users\\louis\\OneDrive\\Desktop\\NN\\good\\test2.pt')
	if epoch < 10 and epoch_acc[epoch+1,:].item() < 0.4:
		print("Network is not learning, change hyperparameters")
		break
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
print("Max loss achieved: ", np.max(epoch_loss))
plt.show()