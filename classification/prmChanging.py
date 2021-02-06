import os,re
pth = os.getcwd().split('\\')
if pth[-1].lower() == 'face':
	os.chdir('classification')
import numpy as np
import matplotlib.pyplot as plt
# 看看各个参数的变化情况
prms = range(0,5512)	# 要显示的参数下标(0-512)
cs = "brkgm"	# 颜色
ms = "^vo><*|"
folder = "backup"
fileList0 = os.listdir(folder)[95:]
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
# for i in range(len(f)-1):
# 	f[i] = f[i][::]
# 	for j in range(len(f[i])):
# 		temp = f[i][j]
# 		temp = temp-f[i][j-1]
# 		f[i][j] = temp

for pi in range(len(prms)):
	i = prms[pi]
	y = f[i]
	x = ts
	plt.ylabel(str(i),fontsize = 14)
	# print(i)
	plt.scatter(x, y, c=cs[pi%len(cs)], s=16, alpha=0.2, marker=ms[pi%len(ms)])
plt.show()