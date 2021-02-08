import os
import torch
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
accFolder = 'accounts'	# 储存账号的文件夹

train_m_folder = "./data/train_m_ft"
train_f_folder = "./data/train_f_ft"
test_m_folder = "./data/test_m_ft"
test_f_folder = "./data/test_f_ft"

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

# I know the name is a biiiiit long, i don't care
def calc_dist_and_show():
	fts = getfts(train_m_folder) + getfts(train_f_folder)

	dist_same = []
	for i in range(0, len(fts), 2):
		f0 = fts[i]
		f1 = fts[i+1]
		dist_same.append(float(torch.sqrt(torch.sum((f1-f0)**2))))
		if dist_same[-1] == 0:
			print(i+1)

	dist_diff = []
	comb = set()
	while len(comb) < 500:
		i = random.randint(0, 498)
		j = random.randint(i+1, 499)
		k = '_'.join([str(i), str(j)])
		comb.add(k)
	for k in comb:
		i, j = map(int, k.split('_'))
		d0 = i*2 + random.randint(0, 1)
		d1 = j*2 + random.randint(0, 1)
		f0 = fts[d0]
		f1 = fts[d1]
		dist_diff.append(float(torch.sqrt(torch.sum((f1 - f0) ** 2))))

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

if __name__ == '__main__':
	calc_dist_and_show()