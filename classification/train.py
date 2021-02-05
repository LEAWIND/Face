import os, re, math, time
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
	f = torch.tensor(f)
	f = torch.cat([f, torch.tensor([1])])
	fts.append(f)
fts = fts[::2]	# 对每个人取 1 张图片就够了
train_m = fts[   :200][:-1]	# 训练用的男性样本
train_f = fts[250:450][:-1]	# 训练用的女性样本
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


# while n>0:
# 	loss_now = loss(p)
# 	# if n%10 == 1:
# 	# 	print(p, 'loss =', loss_now)
# 	# 求偏导, 将结果保存在 d
# 	d = p.clone()
# 	q = p.clone()
# 	for i in range(len(p)):
# 		q[i] += step
# 		d[i] = (loss(q) - loss_now) / step
# 	# 前进
# 	p -= step*d
# 	n -= 1



def move(step=0.00001):	# 每走完一步都把结果存在文件中
	file = open('target.txt', mode='r')
	prm = file.read().strip().split(' ')
	prm = torch.tensor(list(map(float, prm)))
	file.close()

	l0 = lossOf(prm)
	print('loss =', l0)
	d = prm.clone()
	for i in range(len(prm)):
		q = prm.clone()
		q[i] += step
		d[i] = (lossOf(q) - l0) / step

	prm -= step*d

	prm = map(float, prm)
	prm = map(str, prm)
	prm = ' '.join(prm)
	file = open('target.txt', mode='w')
	file.write(prm)
	file.close()
	return 

if __name__ == '__main__':
	n = 2
	while n>0:
		move(0.006)
		n -= 1