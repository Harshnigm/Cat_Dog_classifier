import cv2
import numpy as np
import os
from random import shuffle

train_dir='train'
test_dir='test'
img_size=50
LR=1e-3

model_name='dogsvscats'+str(LR)+'2onv-basic'

def label_img(img):
	 word_label=img.split('.')[-3]
	 if word_label=='cat':
	 	return [1,0]
	 return [0,1]

def create_train_data():
	train_data=[]
	for img in os.listdir(train_dir):
		label=label_img(img)
		path=os.path.join(train_dir,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
		train_data.append([np.array(img),np.array(label)])
	shuffle(train_data)
	print('training_data')
	np.save('train_data.npy',train_data)
	return train_data

def create_test_data():
	test_data=[]
	for img in os.listdir(test_dir):
		path=os.path.join(test_dir,img)
		img_id=img.split('.')[0]
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
		test_data.append([np.array(img),img_id])
	
	np.save('test_data.npy',test_data)
	return test_data



create_test_data()