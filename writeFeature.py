import os, re
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from network.resnet100 import KitModel
class getfacefeature(object):
	def __init__(self):
		model_path = os.path.join('./Arcface_100.pth')  # Parameters path
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

imageFolders = ['./data/train_f', './data/train_m', './data/test_f', './data/test_m']
imageFormats = ['jpg', 'png']

if __name__ == "__main__":
	for fd in imageFolders:
		if os.path.isdir(fd):
			fd_ft = fd + '_ft'
			if not os.path.isdir(fd_ft):
				os.mkdir(fd_ft)
			ims = os.listdir(fd)
			for imN in ims:
				imf = fd +'/'+ imN
				imf_ft = fd_ft +'/'+ imN + '.txt'
				if os.path.isfile(imf) and (imf[-3:] in imageFormats) and not os.path.isfile(imf_ft):
					print(imf_ft)
					im = Image.open(imf)
					ft = ' '.join(map(str, map(float, p.getfeature(im))))
					im.close()
					file = open(imf_ft, mode='w')
					file.write(ft)
					file.close()













exit()
ftFolder = '../data/imgFeature'
imgFolder = '../data/image'
f = ftFolder
# imglist = os.listdir(imgFolder)
imglist = os.listdir("../data/test_f")


for fn in imglist:
	p0 = '/'.join(["../data/test_f", fn])
	p1 = '/'.join([f, fn[:-4]+'.txt'])
	img = Image.open(p0)
	fea = ','.join(map(str, map(float, p.getfeature(img))))
	img.close()
	ftf = open(p1, mode='w')
	ftf.write(fea)
	ftf.close()