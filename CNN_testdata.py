import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import datasets
import random


EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data_all = datasets.MNIST(
	root = './minst',
	train = True,
	transform = torchvision.transforms.ToTensor(),
	download = True)

label=train_data_all.targets
index = train_data_all.data
train_data=train_data_all.data
# print(train_data.shape)


# print(train_data)

def get_img(l):
	index = np.where(label == l)
	img = train_data[index]
	return img



def get_train_data_set():
	img_1 = get_img(1)
	img_2 = get_img(2)
	img_3 = get_img(3)
	img_4 = get_img(4)
	img_5 = get_img(5)

	img = []
	img.append(img_1)
	img.append(img_2)
	img.append(img_3)
	img.append(img_4)
	img.append(img_5)

	train_data = []
	for i in range(5):
		one_img = random.choice(img[i])
		train_data.append(one_img)
	return train_data

