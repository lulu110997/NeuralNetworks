#!/usr/bin/env python3
from os import listdir, pardir
from os.path import join
import cv2
import numpy as np
import torch
from math import ceil as ceil

class PreProcessingData():		
	
	def __init__(self):
		pass

	def SaveAsNumpyArray(self, training_img_path, ratio = 0.8, colour=1, image_size=90, validation_img_path=None):
		'''
		# image_size: Image size for the network input
		# training_img_path: Directory which contains only the folder(s) that the training image files are in
		# validation_img_path: Directory which contains only the folder(s) that the validation images are in
		# ratio: ratio of training to validation split if both image files are in the same folders
		# colour: cv2.imread flag. 0 --> grayscale, 1 --> colour, -1 --> unchanged
		
		A numpy list is saved in the train_img_path which contains a list of images and their corresponding labels.
		Returns the save path for the training and validation path. The save paths will be in the parent directory
		of training_img_path

		IMPORTANT: Make sure that the directory that contains the images for both the training and validation
		have the same name. Otherwise, the label for the training images and validation images will be different 
		'''

		image_train_folders = [] # Populate with the path that contains to the folders which contains the training images
		num_classes = len(listdir(training_img_path)) # Used for specifying how many rows of the identity matrix we need. Used for creating a one hot vector
		pathh = join(training_img_path,pardir) # Obtain the parent directory of the training_img_path	
		if validation_img_path: # If training and validation images are not in the same directory
			img_data = np.zeros((1,2)) # Populate with the image data and their corressponding labels
			validation_img_folders = []
			for d in listdir(training_img_path):
				image_train_folders.append(join(training_img_path,d)) # Obtain the directory path for the training image folders
			for d in listdir(validation_img_path):
				validation_img_folders.append(join(validation_img_path,d)) # Obtain the directory path for the validation image folders

			row = -1 # Used to select the one hot vector in the identity matrix
			for f in image_train_folders: # Iterate through all the folders that contains the training images
				row += 1
				print(f"For {f}, the training label is {np.eye(num_classes)[row]}")
				for i in listdir(f): # Select all the images in the folder
					try:
						img = cv2.imread(join(f,i), colour)
						img = cv2.resize(img,(image_size,image_size))
					except:
						print("Please check the following files are not corrupted ", join(f,i))
					img_data = np.vstack((img_data,np.array([img, np.eye(num_classes)[row]], dtype=object)))
			# Save the training data and its path
			img_data = img_data[1:]
			np.random.shuffle(img_data)
			print("\nPaths to folders that contains the training images include: ", image_train_folders)
			print("\nThe number of training images in this dataset is,", len(img_data))
			save_path = []
			save_path.append(join(pathh, 'train.npy'))
			np.save(save_path[0],img_data)

			# Repeat the above operation but for the VALIDATION images
			img_data = np.zeros((1,2)) # Re-initialise this variable to store validation images
			row = -1 # Used to select the one hot vector in the identity matrix
			for f in validation_img_folders: # Iterate through all the folders that contains the validation images
				row += 1
				print(f"For {f}, the training label is {np.eye(num_classes)[row]}")
				for i in listdir(f): # Select all the images in the folder
					try:
						img = cv2.imread(join(f,i), colour)
						img = cv2.resize(img,(image_size,image_size))
					except:
						print("Please check that the following files are not corrupted ", join(f,i))
					img_data = np.vstack((img_data,np.array([img, np.eye(num_classes)[row]], dtype=object)))
			img_data = img_data[1:]
			np.random.shuffle(img_data)
			print("\nPaths to folders that contains the validation images include: ", validation_img_folders)
			print("\nThe number of validation images in this dataset is,", len(img_data))
			save_path.append(join(pathh, 'validation.npy'))
			np.save(save_path[1],img_data)
			return save_path

		else: # If training and validation images are in the same directory. Will need to split images based on ratio
			train_imgs = np.zeros((1,2)) # Populate with the trainings images and their corressponding labels
			val_imgs = np.zeros((1,2)) # Populate with the validation images and their corressponding labels
			for d in listdir(training_img_path):
				image_train_folders.append(join(training_img_path,d)) # Obtain the directory path for the image folders
			row = -1 # Used to select the one hot vector in the identity matrix
			for f in image_train_folders: # Iterate through all the folders that contains the images
				row += 1
				num_imgs = len(listdir(f)) # Number of images in the folder
				num_train_imgs = ceil(ratio*num_imgs) # Number of required training images
				num_val_imgs = num_imgs-num_train_imgs # Number of required validation images
				for i in listdir(f): # Iterate through all the images in the folder
					while num_train_imgs:
						try:
							img = cv2.imread(join(f,i), colour)
							img = cv2.resize(img,(image_size,image_size))
						except:
							print("Please check the following files are not corrupted ", join(f,i))
						train_imgs = np.vstack((train_imgs,np.array([img, np.eye(num_classes)[row]], dtype=object)))
						num_train_imgs -= 1

					while num_val_imgs:
						try:
							img = cv2.imread(join(f,i), colour)
							img = cv2.resize(img,(image_size,image_size))
						except:
							print("Please check the following files are not corrupted ", join(f,i))
						val_imgs = np.vstack((val_imgs,np.array([img, np.eye(num_classes)[row]], dtype=object)))
						num_val_imgs -= 1

			train_imgs = train_imgs[1:]
			val_imgs = val_imgs[1:]
			np.random.shuffle(train_imgs)
			np.random.shuffle(val_imgs)
			print("\nPaths to folders that contains all the images include: ", image_train_folders)
			print("\nThe number of training images in this dataset is,", len(train_imgs))
			print("\nThe number of validation images in this dataset is,", len(val_imgs))
			save_path = []
			save_path.append(join(pathh, 'train.npy'))
			np.save(save_path[0],train_imgs)
			save_path.append(join(pathh, 'validation.npy'))
			np.save(save_path[1],val_imgs)
			return save_path

	def SaveAsTensor(self, npy_path, images_file_name='image.pt', labels_file_name='labels.pt', image_size=90, image_depth=3):
		'''
		Saves the npy file as a tensor. It uses {images,labels}_path as the name for the pt files
		The pt files are saved in the same directory as the npy path. The absolute for the pt
		files are then returned
		'''
		dataset = np.load(npy_path, allow_pickle=True)
		print("Turning np data as a tensor...\n")
		x = torch.Tensor([i[0] for i in dataset]).view(-1,image_depth,image_size,image_size)
		x = x/255
		y = torch.Tensor([i[1] for i in dataset])
		save_path = []
		pathh = join(npy_path,pardir)
		save_path.append(join(pathh,images_file_name))
		torch.save(x, save_path[0])
		save_path.append(join(pathh,labels_file_name))
		torch.save(y, save_path[1])
		print("Dataset saved as tensor\n")
		return save_path

	def CheckImagesAndLabels(self, train_path, val_path=None):
		'''
		Checks that the labels for the training and validation images match up
		'''
		training_data = np.load(train_path, allow_pickle=True)
		if val_path:
			validation_data = np.load(val_path, allow_pickle=True)
			for i in range(0,len(validation_data), int(len(validation_data)/10)):
				print("training label: ", training_data[i][1])
				print("validation label: ", validation_data[i][1], "\n")
				img_concate_Hori=np.concatenate((training_data[i][0],validation_data[i][0]),axis=1)
				cv2.imshow('Left: validation img, Right: validation img',img_concate_Hori)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
		else:
			for i in range(0,len(training_data), int(len(training_data)/10)):
				print("training label: ", training_data[i][1])
				cv2.imshow('Training data image',training_data[i][0])
				cv2.waitKey(0)
				cv2.destroyAllWindows()

	def CheckTorchDataset(self, train_images_path, train_labels_path, val_images_path=None, val_labels_path=None, image_size=90, image_depth=3):
		'''
		Checks that the labels for the training and validation images match up
		'''
		train_data_x = torch.load(train_images_path)
		train_data_y = torch.load(train_labels_path)
		if val_images_path:
			val_data_x = torch.load(val_images_path)
			val_data_y = torch.load(val_labels_path)
			for i in range(50,len(val_data_x), int(len(val_data_x)/10)):
				print("training label: ", train_data_y[i])
				print("validation label: ", val_data_y[i], "\n")
				img_concate_Hori=np.concatenate(((train_data_x[i].view(image_size,image_size,image_depth)).numpy(),(val_data_x[i].view(image_size,image_size,image_depth)).numpy()),axis=1)
				cv2.imshow('Left: training data, Right: validation data',img_concate_Hori)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
		else:
			for i in range(50,len(train_data_x), int(len(train_data_x)/10)):
				print("training label: ", train_data_y[i])
				cv2.imshow('Training data image:',train_data_x[i])
				cv2.waitKey(0)
				cv2.destroyAllWindows()

##############################
def main():

	inst = PreProcessingData()
	training_img_path = 'C:\\Users\\louis\\Downloads\\del\\testingmodule\\Train'
	validation_img_path = 'C:\\Users\\louis\\Downloads\\del\\testingmodule\\Validate'
	save_paths = inst.SaveAsNumpyArray(training_img_path=training_img_path, image_size=90, ratio=0.5, validation_img_path=validation_img_path)
	train_tensor = inst.SaveAsTensor(save_paths[0], 'training_images.pt', 'training_labels.pt')
	val_tensor = inst.SaveAsTensor(save_paths[1], 'validation_images.pt', 'validation_labels.pt')
	inst.CheckImagesAndLabels(save_paths[0])
	inst.CheckImagesAndLabels(save_paths[1])
	inst.CheckTorchDataset(train_tensor[0], train_tensor[1], val_tensor[0], val_tensor[1])

if __name__ == "__main__":
	main()
