import cv2
import numpy as np
import os
from random import shuffle

train_dir='train'
test_dir=''
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
	number=0
	for img in os.listdir(train_dir):
		label=label_img(img)
		path=os.path.join(train_dir,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
		cv2.imshow('image',img)
		cv2.waitKey(0)
		train_data.append([np.array(img),np.array(label)])
		#print number
		number+=1
		#if number==10:
                 #   break
	shuffle(train_data)
	np.save('train_data.npy',train_data)
	return train_data

create_train_data()

