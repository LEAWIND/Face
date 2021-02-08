import os, re, math, time
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import numpy as np
import matplotlib.pyplot as plt
import torch

train_m_folder = "../data/test_m_ft"
train_f_folder = "../data/test_f_ft"

	
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

fts_m = getfts(train_m_folder)
fts_f = getfts(train_f_folder)

file = open('target.txt', mode='r')
prm = file.read().strip().split(' ')
prm = torch.tensor(list(map(float, prm)))
file.close()

prepre_m = []
pre_m = []
prepre_f = []
pre_f = []

for ft in fts_m:
	tp = torch.sum(prm*ft)
	prepre_m.append(tp)
	pre_m.append(1 / (1 + 2.718281828459045 ** (-tp)))
for ft in fts_f:
	tp = torch.sum(prm*ft)
	prepre_f.append(tp)
	pre_f.append(1 / (1 + 2.718281828459045 ** (-tp)))

# plt.scatter(prepre_m, pre_f, s=20, c="b", alpha=0.2, marker='v')
# plt.scatter(prepre_f, pre_m, s=20, c="r", alpha=0.2, marker='^')
# pre_f = list(map(lambda x: max(pre_m)-x, pre_f))
plt.scatter(prepre_m, pre_m, s=20, c="b", alpha=0.6, marker='v')
plt.scatter(prepre_f, pre_f, s=20, c="r", alpha=0.6, marker='^')
plt.show()