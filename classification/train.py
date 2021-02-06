import os, re, math, time
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import torch
import matplotlib.pyplot as plt
import numpy as np


def isMale(ft:'加了常数项的图像特征', prm:'训练好的513个参数'):
	ft = prm * ft
	ft = torch.sum(ft)
	ft = 1 / (1 + 2.718281828459045 ** (-ft))
	return ft

def lossOf(prm):	# 损失函数
	"""
	共 500 人，取前 200 个男性和前 200 个女性用来训练，剩下 100 个人用来测试
	"""
	ls = 0
	for s in train_m:
		tp = torch.sum(prm*s)
		tp = 1 / (1 + 2.718281828459045 ** (-tp))
		tp = (1-tp) ** 0.5
		ls += tp
	for s in train_f:
		tp = torch.sum(prm*s)
		tp = 1 / (1 + 2.718281828459045 ** (-tp))
		tp = (tp) ** 0.5
		ls += tp
	return ls/len(train_m)


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

if __name__ == '__main__':
		
	ftFolder = '../data/imgFeature'
	ftList = os.listdir(ftFolder)	# 获取文件夹中所有文件名
	fts = []
	for f in ftList:
		fi = open(f"{ftFolder}/{f}", mode='r')
		f = fi.read().split(',')
		fi.close()
		f = list(map(float, f))
		f = torch.tensor(f)
		f = torch.cat([f, torch.tensor([1])])
		fts.append(f)
	fts = fts[::2]	# 对每个人取 1 张图片就够了
	train_m = fts[   :200][:100]	# 训练用的男性样本
	train_f = fts[250:450][:100]	# 训练用的女性样本
	test_m = fts[200:250]	# 测试用的男性样本
	test_f = fts[450:]	# 测试用的女性样本

	n = 1
	while n>0:
		move(0.01)
		print(n)
		n -= 1