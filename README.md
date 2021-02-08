<h1 style="white-space:pre;text-align:center;margin:1em;">   rén  gōng  zhì  zhàng<br/>人      工      智      能</h1>

```python
""" 先看看作业是啥 """
#用feature tensor来计算欧式距离(可以用tensor，也可以转numpy)
#画1000个不同人特征的欧式距离的点和1000个相同人图片的特征的欧式距离
#附加题 用获取的feature 和 label来做分类（参考上一次的逻辑回归）
```

## 写了乱七八糟的好多东西，在这理一理每个文件的作用:

##### [`./writeFeature.py`](./verification/writeFeature.py) 节省时间

把每张图片的特征算出来后直接保存到一些 .txt 文件里。

虽然体积有点大，但是可以加快速度。

### 作业 [0, 1]

* ##### [`./showGraph.py`](./showGraph.py) 显示欧氏距离图像

就是用 matplotlib 把图像画出来嘛

顺便计算 "用来区分是否为同一人" 的阈值，如果欧氏距离大于这个阈值就可以认为不是同一人。



### 作业 [2] 性别分类 classification

先说一下判断性别的原理：

先获取图像特征(长度为 512 的向量)，然后在特征后边加一个 1， 向量长度变成 513，记为 **a**

文件 [`target.txt`](classification/target.txt)  里有 513 个训练好的参数，这些参数组成一个向量，记为 **b**

计算 sigmoid(sum(**a** 点乘 **b**) ) 的值

如果大于 0.5，判断为男性

如果小于 0.5，判断为女性

* ##### [`./classification/train.py`](./classification/train.py) 		训练 (Python版)

* ##### [`./classification/train.java`](classification/train.java) 	训练 (Java版)

python 实在是太慢了，算到虎年都算不完👴 也许因为我的 GPU 是 MX350？

所以我把 train.py 移植到了 [train.java](classification/train.java), 用 [这个bat脚本](classification/compile and run.bat) 编译并运行。每次运行训练的步数是写在 train.java 源码里的。

每训练一定步数，就会把结果写在 [`target.txt`](classification/target.txt) 里。随时可以用它们做测试。

* ##### [`./classification/test.py`](./classification/test.py) 测试

用 `target.txt` 文件中参数做测试，计算这堆参数的准确率（这是用来给人看的，不是训练时用的损失函数）

测试集是我另外找的。

* ##### [./classification/showGraph.py](./classification/showGraph.py) 显示图像

画出 训练集 或 测试集 中图片的性别预测值的图像

* ##### [`backup.py`](classification/backup.py), [`autoBackup.bat`](classification/autoBackup.bat) 备份`target.txt`文件

懒得 ctrl+C ctrl + V，所以用这个脚本每隔`60s`备份一次`target.txt`

就是复制到 [`backups`](classification/backup)文件夹里，然后文件名后边加个时间

* ##### [`./classification/prmChanging.py`](./classification/prmChanging.py) 参数变化趋势

根据 backup 中的数据绘制各参数的值随时间的变化的图像

有这些参数的变化趋势是比较明显的曲线:

`3,6,19,27,59,86,127,134,140,167,182,230,237,241,250,254,255,257,263,293,295,299,309,315,318,322,328,330,333,339,343,367,387,400,403,408,412,413,425,436,444,449,466,470,489,499,512`

备份了一千多次的时候我通过图像发现，很多参数的变化趋势近似抛物线，或者近似双勾函数，那些近似直线的可能是曲线的一部分。

* ##### [`compile and run.bat`](classification/compile and run.bat) 编译并运行`train.java`

只是因为懒得换 IDE

##### [`./classification/judge.py`](./classification/judge.py) 调用摄像头判断性别





### 人脸识别 recognition

* ##### [`./recognition/register.py`](./recognition/register.py) 注册人脸。

如果电脑有摄像头，可以直接运行该文件进行注册。

如果没有，也可以把 图像.jpg(.png...) 放在文件夹[`./recognition/account`](./recognition/account)里，然后直接运行`register.py`。

* ##### [`./recognition/recognize.py`](./recognition/recognize.py) 通过摄像头识别人脸

通过摄像头实时获取图像，提取特征，与文件夹`account`里的`*.txt`中的特征相比对，找出最相似的人

原理就是计算特征的欧氏距离。



```js
'有时用 CPU 运行会有这样的报错：';`
Traceback (most recent call last):
...
RuntimeError: [enforce fail at ..\c10\core\CPUAllocator.cpp:73] data. DefaultCPUAllocator: not enough memory: you tried to allocate 9437184 bytes. Buy new RAM!
...
`
'它竟然叫我买新内存条 ! ? ?'

```



