# 性别分类训练
import os, re, math, time
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import torch
import matplotlib.pyplot as plt
import numpy as np



train_m_folder = "../data/train_m_ft"
train_f_folder = "../data/train_f_ft"

def lossOf(prm):	# 损失函数
	ls = 0


	for s in train_m:


		tp = torch.sum(prm*s)

		ls += (1 - 1 / (1 + 2.718281828459045 ** (-tp)))

	for s in train_f:


		tp = torch.sum(prm*s)

		ls += 1 / (1 + 2.718281828459045 ** (-tp))

	return ls / len(train_m)

def move(step=0.0001):
	global prm
	loss_now = lossOf(prm)
	d = prm.clone()
	for i in range(len(prm)):	# 对每个方向分别求导
		q = prm.clone()
		q[i] += step
		d[i] = (lossOf(q) - loss_now) / step

	prm = map(str, map(float, prm - step*d))


	print('loss =', loss_now)


	return

def readPrm():
	global prm

	file = open('target.txt', mode='r')

	prm = file.read().strip().split(' ')

	prm = torch.tensor(list(map(float, prm)))

	file.close()



	return prm

def savePrm(prm):

	prm = ' '.join(prm)




	file = open('target.txt', mode='w')
	file.write(prm)

	file.close()


def getfts(folderPath):

	ftList = os.listdir(folderPath)
	fts = []
	for f in ftList:



		fi = open(f"{folderPath}/{f}", mode='r')


		f = (fi.read()+' 1').split(' ')

		fi.close()


		f = torch.tensor(list(map(float, f)))

		fts.append(f)

	return fts






prm = []
if __name__ == '__main__':

	train_f = getfts(train_f_folder)
	train_m = getfts(train_m_folder)

	train_m = train_m[102:500]
	train_f = train_f[102:400]

	prm = readPrm()
	n = 1
	# n = 2
	while n>0:
		n -= 1
		move(0.02)
		if n % 20 == 0:
			print(n)
			# savePrm(prm)

	# savePrm(prm)

