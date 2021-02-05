import os, re
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

ftFolder = '../data/imgFeature'
ftList = os.listdir(ftFolder)
fts = []
for f in ftList:
	fi = open(f"{ftFolder}/{f}", mode='r')
	f = fi.read().split(',')
	fi.close()
	f = list(map(float, f))
	f = torch.tensor(f)
	f = torch.cat([f, torch.tensor([1])])	# 加个常数 1 上去
	fts.append(f)

def isMale(ft:'加了常数项的图像特征', prm:'训练好的513个参数'):
	ft = prm * ft
	ft = torch.sum(ft)
	ft = 1 / (1 + 2.718281828459045 ** (-ft))
	return ft



if __name__ == '__main__':
	file = open('target.txt', mode='r')
	prm = file.read().strip().split(' ')
	prm = torch.tensor(list(map(float, prm)))
	file.close()
	print( isMale(fts[0], prm) )