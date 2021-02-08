import os, re, math, time
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import torch
import matplotlib.pyplot as plt
import numpy as np

train_m_folder = "../data/train_m_ft"
train_f_folder = "../data/train_f_ft"

def isMale(ft:'加了常数项的图像特征', prm:'训练好的513个参数'):
	ft = prm * ft
	ft = torch.sum(ft)
	ft = 1 / (1 + 2.718281828459045 ** (-ft))
	return ft

def lossOf(prm):	# 损失函数
	ls = 0
	for s in train_m:
		tp = torch.sum(prm*s)
		tp = 1 - 1 / (1 + 2.718281828459045 ** (-tp))
		ls += tp
	for s in train_f:
		tp = torch.sum(prm*s)
		tp = 1 / (1 + 2.718281828459045 ** (-tp))
		ls += tp
	return ls / len(train_m)


def move(step=0.0001):	# 每走完一步都把结果存在文件中
	file = open('target.txt', mode='r')
	prm = file.read().strip().split(' ')
	prm = torch.tensor(list(map(float, prm)))
	file.close()

	loss_now = lossOf(prm)
	d = prm.clone()
	for i in range(len(prm)):
		q = prm.clone()
		q[i] += step
		d[i] = (lossOf(q) - loss_now) / step

	prm = map(float, prm - step*d)
	prm = map(str, prm)
	prm = ' '.join(prm)
	file = open('target.txt', mode='w')
	file.write(prm)
	file.close()
	print('loss =', loss_now)
	return
def getfts(folderPath):
	ftList = os.listdir(folderPath)
	fts = []
	for f in ftList:
		fi = open(f"{folderPath}/{f}", mode='r')
		f = fi.read().split(' ')
		fi.close()
		f = list(map(float, f))
		f = torch.tensor(f)
		f = torch.cat([f, torch.tensor([1])])
		fts.append(f)
	return fts

if __name__ == '__main__':
	train_m = getfts(train_m_folder)	# 训练用的男性样本
	train_f = getfts(train_f_folder)	# 训练用的女性样本
	n = 1
	while n>0:
		move(0.01)
		print(n)
		n -= 1