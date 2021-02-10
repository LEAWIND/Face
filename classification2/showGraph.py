import os, re, math, time
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification2')
import numpy as np
import matplotlib.pyplot as plt
import torch

s = ''
while s == '':
	i = input("你想看训练集(0)还是测试集(1)？")
	s = "test" if i=='1' else "train" if i=='0' else ''
train_m_folder = f"../data/{s}_m_ft"
train_f_folder = f"../data/{s}_f_ft"
	
def getfts(folderPath):
	ftList = os.listdir(folderPath)
	fts = []
	for f in ftList:
		fi = open(f"{folderPath}/{f}", mode='r')
		f = fi.read().split(' ')
		fi.close()
		f = list(map(float, f))
		f = torch.tensor(f)
		f= torch.cat([f**2, f, torch.tensor([1])])
		fts.append(f)
	return fts

fts_m = getfts(train_m_folder)
fts_f = getfts(train_f_folder)

file = open('target.txt', mode='r')
prm = file.read().strip().split(' ')
prm = torch.tensor(list(map(float, prm)))
file.close()

pre_m = []
pre_f = []

for ft in fts_m:
	tp = torch.sum(prm*ft)
	pre_m.append(tp)
for ft in fts_f:
	tp = torch.sum(prm*ft)
	pre_f.append(tp)

leng = min(len(pre_f), len(pre_m))
pre_m = pre_m[:leng]
pre_f = pre_f[:leng]

plt.scatter(pre_m, pre_f, s=20, c="b", alpha=0.5, marker='v')
plt.scatter(pre_f, pre_m, s=20, c="r", alpha=0.5, marker='^')

plt.scatter(pre_m, pre_m, s=20, c="b", alpha=0.3, marker='v')
plt.scatter(pre_f, pre_f, s=20, c="r", alpha=0.3, marker='^')
plt.show()