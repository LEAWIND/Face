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
train_m = fts[   :200][:20]	# 训练用的男性样本
train_f = fts[250:450][:20]	# 训练用的女性样本
test_m = fts[200:250]	# 测试用的男性样本
test_f = fts[450:]	# 测试用的女性样本
