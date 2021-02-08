import os
import torch
import matplotlib.pyplot as plt
import random

train_m_folder = "./data/train_m_ft"
train_f_folder = "./data/train_f_ft"

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

def xloss(x):
	cor = 0
	err = 0
	for i in dist_same:
		if x > i:
			cor += 1
		elif x < i:
			err += 1
	for i in dist_diff:
		if x < i:
			cor += 1
		elif x > i:
			err += 1
	return cor / (cor + err), cor, err

if __name__ == '__main__':
	fts = getfts(train_m_folder) + getfts(train_f_folder) # 不得不说 python 是真的方便
	# 相同人的欧式距离
	dist_same = []
	for i in range(0, len(fts), 2):
		f0 = fts[i]
		f1 = fts[i+1]
		dist_same.append(float(torch.sqrt(torch.sum((f1-f0)**2))))
		if dist_same[-1] == 0:
			print(i+1)
	# 不同人的欧式距离
	dist_diff = []
	comb = set()
	while len(comb) < 500:
		i = random.randint(0, 498)
		j = random.randint(i+1, 499)
		comb.add('_'.join([str(i), str(j)]))
	for k in comb:
		i, j = map(int, k.split('_'))
		f0 = fts[i*2 + random.randint(0, 1)]
		f1 = fts[j*2 + random.randint(0, 1)]
		dist_diff.append(float(torch.sqrt(torch.sum((f1 - f0) ** 2))))
	# 找阈值
	x = 0.8
	step = 0.01
	n = 2000
	while n > 0:
		a = xloss(x + step)[0]
		b = xloss(x - step)[0]
		if a > b:
			x += 0.001
		elif a < b:
			x -= 0.001
		n -= 1
	n = xloss(x)
	print(f"取阈值为 {round(x, 3)} 时的准确率约为: \n {n[1]} / {n[1]+n[2]} = {n[0]}")

	plt.scatter(dist_same, dist_same, c='b', marker='o', alpha=0.2, s=10)
	plt.scatter(dist_diff, dist_diff, c='r', marker='o', alpha=0.2, s=10)
	plt.scatter(dist_same, dist_diff, c='b', marker='o', alpha=0.2, s=10)
	plt.scatter(dist_diff, dist_same, c='r', marker='o', alpha=0.2, s=10)
	plt.show()