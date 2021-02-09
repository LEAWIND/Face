# 各个参数随时间的变化图像

import os,re
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import numpy as np
import matplotlib.pyplot as plt

# 参数们的下标:
prms = [3,59,86,134,167,230,237,241,254,255,263,295,299,309,315,318,322,328,330,333,339,343,367,387,400,403,408,412,413,425,436,466,470,489,499,512]	# 要显示的参数下标(0-512)
# prms = range(0,513)
cs = "brgm"	# 颜色
ms = "."	# ^o*>v<|

folder = "backup"

fileList0 = os.listdir(folder)[-1000::1]
fileList = []
for i in fileList0:
	if i[-4:] == '.txt':
		fileList.append(i)
zps = []
ts =  []
for fn in fileList:
	ts.append(int(re.sub(r'[^0-9]', '', fn)))
	file = open(f"{folder}/{fn}")
	fn = file.read().strip().split(' ')
	fn = list(map(float, fn))
	zps.append(fn)
f = []
i = 0
while i < len(zps[0]):
	f.append([])
	for j in zps:
		f[i].append(j[i])
	i += 1

for pi in range(len(prms)):
	i = prms[pi]
	y = f[i]
	x = ts
	plt.ylabel(str(i),fontsize = 8)
	print(i)
	plt.scatter(x, y, c=cs[pi%len(cs)], s=0.5, alpha=0.8, marker=ms[pi%len(ms)])
plt.show()