#!/usr/bin/env python3
from os import listdir
from os.path import join
import cv2
import numpy as np
import torch

class PreProcessingData():		
	
	def __init__(self):
		pass

	def SaveAsNumpyArray(self, data_path, save_path, flag = 0, image_size = 128):
		# image_size: Image size for the network input
		# data_path: Directory which contains only the folder(s) that the image files are in
		# save_path: Directory that the npy file will be saved in
		# flag: cv2.imread flag. 0 --> grayscale, 1 --> colour, -1 --> unchanged
		image_folders = [] # Populate with the path that contains to the folders which contains the image
		training_data = np.zeros((1,2)) # Populate with the image data and their corressponding labels
		num_classes = len(listdir(data_path)) # Used for specifying how many rows of the identity matrix we need. Used for creating a one hot vector

		for d in listdir(data_path):
			image_folders.append(join(data_path,d)) # Obtain the directory path for the image folders
		row = -1 # Used to select the one hot vector in the identity matrix
		test_counter = 0 # test
		for f in image_folders: # Iterate through all the folders that contains the images
			row += 1
			for i in listdir(f): # Select all the images in the folder
				try:
					img = cv2.imread(join(f,i), flag)
					img = cv2.resize(img,(image_size,image_size))
				except:
					print("Please check the following files are not corrupted ", join(f,i))
				training_data = np.vstack((training_data,np.array([img, np.eye(num_classes)[row]])))
				test_counter+=1 # test
				if test_counter > 800: # test
					break # test
			test_counter = 0 # test
		training_data = training_data[1:]
		np.random.shuffle(training_data)
		print("\nPaths to folders that contains the images include: ", image_folders)
		print("\nThe number of images in this dataset is,", len(training_data))
		np.save(save_path,training_data)

	def SaveAsTensor(self, npy_path, images_path, labels_path, image_size, image_depth):
		dataset = np.load(npy_path, allow_pickle=True)
		print("Turning np data as a tensor...\n")
		x = torch.Tensor([i[0] for i in dataset]).view(-1,image_depth,image_size,image_size)
		x = x/255
		y = torch.Tensor([i[1] for i in dataset])
		torch.save(x, images_path)
		torch.save(y, labels_path)
		print("Dataset saved as tensor\n")

	def CheckImagesAndLabels(self, train_path, test_path):
		training_data = np.load(train_path, allow_pickle=True)
		testing_data = np.load(test_path, allow_pickle=True)
		for i in range(0,len(testing_data), int(len(testing_data)/10)):
			print("training label: ", training_data[i][1])
			print("testing label: ", testing_data[i][1], "\n")
			img_concate_Hori=np.concatenate((training_data[i][0],testing_data[i][0]),axis=1)
			cv2.imshow('Left: training data, Right: testing data',img_concate_Hori)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	def CheckTorchDataset(self, train_images_path, train_labels_path, test_images_path, test_labels_path, image_size, image_depth):
		train_data_x = torch.load(train_images_path)
		train_data_y = torch.load(train_labels_path)
		test_data_x = torch.load(test_images_path)
		test_data_y = torch.load(test_labels_path)
		for i in range(50,len(test_data_x), int(len(test_data_x)/10)):
			print("training label: ", train_data_y[i])
			print("testing label: ", test_data_y[i], "\n")
			img_concate_Hori=np.concatenate(((train_data_x[i].view(image_size,image_size,image_depth)).numpy(),(test_data_x[i].view(image_size,image_size,image_depth)).numpy()),axis=1)
			cv2.imshow('Left: training data, Right: testing data',img_concate_Hori)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

##############################
def main():
	preProcInst = PreProcessingData()
	preProcInst.SaveAsNumpyArray(data_path='C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\train', save_path='C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD50files\\CD50filestrain.npy', flag=0, image_size=50)
	preProcInst.SaveAsNumpyArray(data_path='C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\test', save_path='C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD50files\\CD50filestest.npy', flag=0, image_size=50)
	npy_trainingpath = 'C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD50files\\CD50filestrain.npy'
	npy_testingpath = 'C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD50files\\CD50filestest.npy'
	training_imagespath = 'C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD50files\\AssignmentTwoTrainingImages.pt'
	training_labelspath = 'C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD50files\\AssignmentTwoTrainingLabels.pt'
	testing_imagespath = 'C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD50files\\AssignmentTwoTestingImages.pt'
	testing_labelspath = 'C:\\Users\\louis\\OneDrive\\Desktop\\NN\\assignment2\\CD50files\\AssignmentTwoTestingLabels.pt'
	preProcInst.SaveAsTensor(npy_testingpath, testing_imagespath, testing_labelspath, 50, 1)
	preProcInst.SaveAsTensor(npy_trainingpath, training_imagespath, training_labelspath, 50, 1)
	preProcInst.CheckImagesAndLabels(npy_trainingpath, npy_testingpath)
	preProcInst.CheckTorchDataset(training_imagespath, training_labelspath, testing_imagespath, testing_labelspath, 50, 1)
	

if __name__ == "__main__":
	main()
