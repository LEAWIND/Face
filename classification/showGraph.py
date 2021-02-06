import os, re, math, time
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import numpy as np
import matplotlib.pyplot as plt
import torch

ftFolder = '../data/imgFeature'
ftList = os.listdir(ftFolder)
fts = []
for f in ftList:
	fi = open(f"{ftFolder}/{f}", mode='r')
	f = fi.read().split(',')
	fi.close()
	f = list(map(float, f))
	f = torch.tensor(f)
	f = torch.cat([f, torch.tensor([1])])
	fts.append(f)
# fts = fts[::2]
fts_m = fts[   :500]
fts_f = fts[500:   ]

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

plt.scatter(prepre_m, pre_f, s=20, c="b", alpha=0.2, marker='v')
plt.scatter(prepre_f, pre_m, s=20, c="r", alpha=0.2, marker='^')
# pre_f = list(map(lambda x: max(pre_m)-x, pre_f))
plt.scatter(prepre_m, pre_m, s=20, c="b", alpha=0.2, marker='v')
plt.scatter(prepre_f, pre_f, s=20, c="r", alpha=0.2, marker='^')
plt.show()