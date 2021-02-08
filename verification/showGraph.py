import os
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('recognition')
import torch
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
accFolder = 'accounts'	# 储存账号的文件夹

train_m_folder = "../data/train_m_ft"
train_f_folder = "../data/train_f_ft"
test_m_folder = "../data/test_m_ft"
test_f_folder = "../data/test_f_ft"

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

	plt.scatter(dist_same, dist_same, c='b', marker='o', alpha=0.3)
	plt.scatter(dist_diff, dist_diff, c='r', marker='o', alpha=0.3)

	plt.scatter(dist_same, dist_diff, c='b', marker='o', alpha=0.3)
	plt.scatter(dist_diff, dist_same, c='r', marker='o', alpha=0.3)
	plt.show()

if __name__ == '__main__':
	calc_dist_and_show()