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

