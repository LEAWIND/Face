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
train_m = fts[   :200]	# 训练用的男性样本
train_f = fts[250:450]	# 训练用的女性样本
test_m = fts[150:250]	# 测试用的男性样本
test_f = fts[400:]	# 测试用的女性样本

# test_m = train_m
# test_f = train_f

def calcAccuracy(prm):
	er = 0
	co = 0
	co_m = 0
	avg_m = 0
	avg_f = 0

	for s in test_m:
		tp = torch.sum(prm*s)
		tp = 1 / (1 + 2.718281828459045 ** (-tp))
		# print('tp =', tp)
		avg_m += tp
		if tp > 0.5:
			co += 1
			co_m += 1
		else:
			er += 1
	
	for s in test_f:
		tp = torch.sum(prm*s)
		tp = 1 / (1 + 2.718281828459045 ** (-tp))
		avg_f += tp
		# print('tp =', tp)
		if tp < 0.5:
			co += 1
		else:
			er += 1
	print(f"总准确率:{co}/{co+er} = {co/(co+er)}")
	print(f"男性:	{co_m}/{len(test_m)} = {co_m/len(test_m)}")
	print(f"女性:	{co-co_m}/{len(test_m)} = {(co-co_m)/len(test_m)}")
	print(f"男性均值:{avg_m/len(test_m)}\n女性均值:{avg_f/len(test_m)}")
	return co / (co + er), co, er


if __name__ == '__main__':
	file = open('target.txt', mode='r')
	prm = file.read().strip().split(' ')
	prm = torch.tensor(list(map(float, prm)))
	file.close()
	calcAccuracy(prm)