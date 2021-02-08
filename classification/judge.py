print('请稍等...')
import os, re
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from network.resnet100 import KitModel
import cv2
prmPath = 'target.txt'

class getfacefeature(object):
	def __init__(self):
		model_path = os.path.join('../Arcface_100.pth')  # Parameters path
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Computing devices
		self.device = torch.device('cpu')	#TODO
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
	ft = torch.cat([ft, torch.tensor([1])])
	file = open(prmPath, mode='r')
	prm = file.read().strip().split(' ')
	file.close()
	prm = torch.tensor(list(map(float, prm)))#.to(p.device)
	preV = torch.sum(prm * ft)
	V = 1 / (1 + 2.718281828459045 ** (-preV))
	V = float(V)
	if V > 0.5:
		print(f"男 {V}")
	elif 0.4<V<0.6:
		print(f"中 {V}")
	else:
		print(f"女 {V}")
