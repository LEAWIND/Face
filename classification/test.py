import os, re, math, time
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import torch

test_m_folder = "../data/test_m_ft"
test_f_folder = "../data/test_f_ft"
test_m_list = os.listdir(test_m_folder)
test_f_list = os.listdir(test_f_folder)

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

def calcAccuracy(prm):
	er = 0
	co = 0
	co_m = 0
	avg_m = 0
	avg_f = 0

	print('m:')
	i = 0
	for s in test_m:
		tp = torch.sum(prm*s)
		tp = 1 / (1 + 2.718281828459045 ** (-tp))
		avg_m += tp
		if tp > 0.5:
			co += 1
			co_m += 1
		else:
			# print(test_m_list[i], tp)
			er += 1
		i += 1
	print('\nf:')
	i = 0
	for s in test_f:
		tp = torch.sum(prm*s)
		tp = 1 / (1 + 2.718281828459045 ** (-tp))
		avg_f += tp
		if tp < 0.5:
			co += 1
		else:
			# print(test_f_list[i], tp)
			er += 1
		i += 1

	print(f"总准确率:{co}/{co+er} = {co/(co+er)}")
	print(f"男性:	{co_m}/{len(test_m)} = {co_m/len(test_m)}")
	print(f"女性:	{co-co_m}/{len(test_f)} = {(co-co_m)/len(test_f)}")
	print(f"男性均值:{avg_m/len(test_m)}	= 1 - {1-avg_m/len(test_m)}")
	print(f"女性均值:{avg_f/len(test_m)}	= 1 - {1-avg_f/len(test_m)}")
	return co / (co + er), co, er


if __name__ == '__main__':
	test_m = getfts(test_m_folder)
	test_f = getfts(test_f_folder)

	file = open('target.txt', mode='r')
	prm = file.read().strip().split(' ')
	prm = torch.tensor(list(map(float, prm)))
	file.close()
	calcAccuracy(prm)