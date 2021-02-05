print("请稍等...")
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
		return feature[0] # tensor
p = getfacefeature()
# 将文件夹中的jpg文件转换成特征值文件
fs = os.listdir(accFolder)
for f in fs:
	if not f[-3:].lower() in ['jpg', 'png', 'gif', 'bmp']:
		continue
	else:
		img = Image.open(f'{accFolder}/{f}')
		fea = p.getfeature(img)
		fea = map(float, fea)
		fea = map(str, fea)
		fea = ','.join(fea)
		img.close()
		file = open(f'{accFolder}/{f[:-4]}.txt', mode='w')
		file.write(fea)
		file.close()

ret = False
while not ret:
	camIdx = input('请输入摄像头设备号(默认为 0)\n/').strip()
	camIdx = 0 if camIdx=='' else int(camIdx)
	cap = cv2.VideoCapture(camIdx)	# 获取摄像头
	ret = cap.read()[0]
	if not ret:
		print('设备号有误')
for i in range(20):
	frameBef = cap.read()
print('摄像头已准备就绪')

accName = 0
while 1:
	tempName = accName
	accName = input("请输入账号名，正视摄像头，准备好后按回车注册。\n/").strip()
	if accName == '':
		if tempName == 0:
			accName = 0
			continue
		print(f'输入的值不合法,取上一次的值: {tempName}')
		accName = tempName
	a_path = f'{accFolder}/{accName}.txt'
	img = cap.read()[1]
	print('已获取图片，正在提取特征...')
	fea = p.getfeature(img)
	if os.path.exists(a_path):
		file = open(a_path, mode='r')
		fea0 = file.read().split(',')
		file.close()
		fea0 = torch.tensor(list(map(float, fea0))).to(p.device)
		fea = (fea + fea0)/2
	fea = map(float, fea)
	fea = map(str, fea)
	fea = ','.join(fea)
	file = open(a_path, mode='w')
	file.write(fea)
	file.close()
	print('保存成功:', f'{accFolder}/{accName}.txt\n')
