import os, re, math
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

ftFolder = '../data/imgFeature'
ftList = os.listdir(ftFolder)
# 把文件里的特征数据读到内存里，这样应该更快
fts = []
for f in ftList:
	fi = open(f"{ftFolder}/{f}", mode='r')
	f = fi.read().split(',')
	fi.close()
	f = list(map(float, f))
	f = torch.tensor(f).to('cuda')
	f = torch.cat([f, torch.tensor([1]).to('cuda')])
	fts.append(f)
fts = fts[::2]	# 对每个人取 1 张图片就够了
train_m = fts[   :200]	# 训练用的男性样本
train_f = fts[250:450]	# 训练用的女性样本
test_m = fts[200:250]	# 测试用的男性样本
test_f = fts[450:]	# 测试用的女性样本

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
		tp = math.log(1-tp)
		ls -= tp
	for s in train_f:
		tp = torch.sum(prm*s)
		tp = 1 / (1 + 2.718281828459045 ** (-tp))
		tp = math.log(tp)
		ls -= tp
	return ls
def move(step=0.00001):	# 每走完一步都把结果存在文件中
	file = open('target.txt', mode='r')
	prm = file.read().strip().split(' ')
	prm = torch.tensor(list(map(float, prm))).to('cuda')
	file.close()
	l0 = lossOf(prm)
	return 

if __name__ == '__main__':
	n = 4
	while n>0:
		move(0.00001)
		n -= 1