print("稍等...")
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
		return feature[0]
p = getfacefeature()

ret = False
while not ret:
	camIdx = input('请输入摄像头设备号\n/').strip()
	camIdx = 0 if camIdx=='' else int(camIdx)
	cap = cv2.VideoCapture(camIdx)	# 获取摄像头
	ret = cap.read()[0]
	if not ret:
		print('设备号有误')
print('摄像头准备就绪')


while 1:
	img = cap.read()[1]
	ft = p.getfeature(img)
	accs = os.listdir(accFolder)
	mb = []
	for accName in accs:
		if not accName[-3:] == 'txt':
			continue
		accName = accName[:-4]
		file = open(f'{accFolder}/{accName}.txt', mode='r')
		ft1 = file.read().split(',')
		file.close()
		ft1 = torch.tensor(list(map(float, ft1))).to(p.device)
		dist = torch.sqrt(torch.sum((ft1-ft)**2))
		if(dist > 1.10):
			continue
		else:
			mb.append([accName, dist])
	if len(mb) == 0:
		print('找不到匹配者')
		continue
	else:
		minFt = mb[0]
		i = 1
		while i < len(mb):
			if mb[i][1] < minFt[1]:
				minFt = mb[i]
			i += 1
		print(f'{minFt[0]}, dist = {minFt[1]}')