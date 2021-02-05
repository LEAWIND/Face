import os
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('recognition')
import torch
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
accFolder = 'accounts'	# 储存账号的文件夹


def calc_dist_and_show():
	ftFolder = '../data/imgFeature'
	ftList = os.listdir(ftFolder)

	dist_same = []
	# 同一人 500 对
	for i in range(0, 1000, 2):
		f0 = '/'.join([ftFolder, ftList[i]])
		file = open(f0, mode='r')
		f0 = torch.tensor(list(map(float, file.read().split(','))))
		file.close()
		f1 = '/'.join([ftFolder, ftList[i+1]])
		file = open(f1, mode='r')
		f1 = torch.tensor(list(map(float, file.read().split(','))))
		file.close()
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
		# print(i, j)
		d0 = i*2+random.randint(0,1)
		d1 = j*2+random.randint(0,1)
		# print(d0, d1)
		f0 = '/'.join([ftFolder, ftList[d0]])
		file = open(f0, mode='r')
		f0 = torch.tensor(list(map(float, file.read().split(','))))
		file.close()

		f1 = '/'.join([ftFolder, ftList[d1]])
		file = open(f1, mode='r')
		f1 = torch.tensor(list(map(float, file.read().split(','))))
		file.close()

		dist_diff.append(float(torch.sqrt(torch.sum((f1-f0)**2))))

	print('相同人特征值的欧氏距离平均值:', sum(dist_same)/len(dist_same))
	print('不同人特征值的欧氏距离平均值:', sum(dist_diff)/len(dist_diff))
	def lossOf(threshold):
		n = 0
		for i in dist_same:
			if i > threshold:
				n += 1
		for i in dist_diff:
			if i < threshold:
				n += 1
		return n
	ts = 0.6
	step = 0.01
	n = 1000
	while n>0:
		alt0 = lossOf(ts-step)
		alt1 = lossOf(ts+step)
		if alt0 < alt1:
			ts -= step
		elif alt0 > alt1:
			ts += step
		else:
			ts += (random.random()*4-2)*step
		n -= 1
	
	print('一个比较合适的阈值:', ts)
	print('它的准确率:', f"Accuracy of({ts}) = 1 - {lossOf(ts)} / 1000 = {1-lossOf(ts)/1000}")

	plt.scatter(dist_same, dist_same, s=30, c='b', alpha=0.2, marker='o')
	plt.scatter(dist_diff, dist_diff, s=30, c='r', alpha=0.2, marker='o')
	plt.scatter(dist_same, dist_diff, s=30, c='b', alpha=0.3, marker='^')
	plt.scatter(dist_diff, dist_same, s=30, c='r', alpha=0.3, marker='v')
	plt.show()




if __name__ == '__main__':
	calc_dist_and_show()
	os.system('pause')