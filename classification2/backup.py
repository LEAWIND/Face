from shutil import *
import time as T
t = T.time()*1000
t = int(t)
t = str(t)
print(t)
tf = 'backup/target-' + t + '.txt'
copy('target.txt', tf)