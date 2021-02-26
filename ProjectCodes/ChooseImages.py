from os.path import join
import numpy as np
import cv2
import random

def ChooseImages(orig_path, new_path, req_imgs, imgs_sample, new_img_name):
	'''
	Simple func that takes out a random amount of images in each 
	folder and saves it in another folder. The saved images are 
	jpg file
	
	orig_path: The directory that contains all the parent directory of
	the image directories
	orig_path
	--> dir1 has imgs1
	--> dir2 has imgs2

	new_path: The directory to save the images in
	req_imgs: The specific directory you want to take imgs out of
	imgs_sample: The number of images you want to take out of the dir
	new_img_name: The base name of the images to be saved
	'''
	orig_folder = orig_path
	img_folders = listdir(orig_folder)
	imgs = []
	apple_paths = []

	for f in img_folders:
		if req_imgs in f:
			apple_paths.append(join(orig_folder,f))

	for p in apple_paths:
		name_imgs = listdir(p)
		num_imgs = len(name_imgs)
		idx = random.sample(range(1, num_imgs), imgs_sample)
		for i in idx:
			imgs.append(cv2.imread(join(p,name_imgs[i]),-1))

	new_folder = new_path
	count = 0
	np.random.shuffle(imgs)
	for i in imgs:
		count += 1
		new_path = new_folder + new_img_name + str(count) + '.jpg'
		cv2.imwrite(new_path,i)