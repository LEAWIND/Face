print("初始化中...")
import os, re, hashlib
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('recognition')
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from network.resnet100 import KitModel
import cv2

import random
accFolder = 'accounts'	# 储存账号的文件夹
	
class getfacefeature(object):
	def __init__(self):
		model_path = os.path.join('../Arcface_100.pth')  # Parameters path
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Computing devices
		self.arcface = KitModel(model_path).to(self.device)  # model
		self.arcface.eval()
		self.transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (128 / 255., 128 / 255., 128 / 255.))
		])
	def getfeature(self, img:"img from PIL.Image.open"):
		img = self.transforms(img).unsqueeze(0)
		img = torch.cat([img, img], dim=0).to(self.device)
		feature = self.arcface(F.interpolate(img, (112, 112), mode='bilinear', align_corners=True))
		return feature[0] # tensor
p = getfacefeature()
print('初始化完成')

def calc_dist_and_show():
	ftFolder = '../data/imgFeature'
	ftList = os.listdir(ftFolder)

	# imgFolder = '../data/image'
	# f = ftFolder
	# imglist = os.listdir(imgFolder)
	# for fn in imglist:
	# 	p0 = '/'.join([imgFolder, fn])
	# 	p1 = '/'.join([f, fn[:-4]+'.txt'])
	# 	img = Image.open(p0)
	# 	fea = ','.join(map(str, map(float, p.getfeature(img))))
	# 	img.close()
	# 	ftf = open(p1, mode='w')
	# 	ftf.write(fea)
	# 	ftf.close()

	dist_same = []
	# 同一人 500 对
	for i in range(0, 1000, 2):
		f0 = '/'.join([ftFolder, ftList[i]])
		file = open(f0, mode='r')
		f0 = torch.tensor(list(map(float, file.read().split(',')))).to(p.device)
		file.close()
		f1 = '/'.join([ftFolder, ftList[i+1]])
		file = open(f1, mode='r')
		f1 = torch.tensor(list(map(float, file.read().split(',')))).to(p.device)
		file.close()
		dist_same.append(float(torch.sqrt(torch.sum((f1-f0)**2))))

	dist_diff = []
	comb = set()
	while len(comb) < 500:
		i = random.randint(0, 498)
		j = random.randint(i+1, 499)
		k = '_'.join([str(i), str(j)])
		comb.add(k)

	for k in comb:
		i, j = map(int, k.split('_'))
		# print(i, j)
		d0 = i*2+random.randint(0,1)
		d1 = j*2+random.randint(0,1)
		# print(d0, d1)
		f0 = '/'.join([ftFolder, ftList[d0]])
		file = open(f0, mode='r')
		f0 = torch.tensor(list(map(float, file.read().split(',')))).to(p.device)
		file.close()

		f1 = '/'.join([ftFolder, ftList[d1]])
		file = open(f1, mode='r')
		f1 = torch.tensor(list(map(float, file.read().split(',')))).to(p.device)
		file.close()

		dist_diff.append(float(torch.sqrt(torch.sum((f1-f0)**2))))


	plt.xlim(min(dist_same), max(dist_diff))
	plt.ylim(min(dist_same), max(dist_diff))
	plt.scatter(dist_same, dist_same,s=10, c='b', alpha=0.2, marker='.')
	plt.scatter(dist_diff, list(map(lambda x: max(dist_diff)-x, dist_diff)),s=10, c='r', alpha=0.2, marker='.')
	plt.show()




if __name__ == '__main__':
	calc_dist_and_show()
