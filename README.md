<h1 style="white-space:pre">rén  gōng  zhì  zhàng<br/>人      工      智      能</h1>

```python
""" 先看看作业是啥 """
#用feature tensor来计算欧式距离(可以用tensor，也可以转numpy)
#画1000个不同人特征的欧式距离的点和1000个相同人图片的特征的欧式距离
#附加题 用获取的feature 和 label来做分类（参考上一次的逻辑回归）
```

## 写了乱七八糟的好多东西，在这理一理:

### [0, 1]

##### [`./verification/writeFeature.py`](./verification/writeFeature.py) 节省时间

把每张图片的特征算出来后直接保存到一些 .txt 文件里。

虽然体积有点大，但是可以加快速度。

##### [`./verification/showGraph.py`](./verification/showGraph.py) 显示欧氏距离图像

就是用 matplotlib 把图像画出来嘛

顺便用类似于梯度下降的方法计算 "用来区分是否为同一人" 的阈值，如果欧氏距离大于这个阈值就认为不是同一人

### [2] 性别分类 classification

先说一下判断性别的原理：

先获取图像特征(长度为 512 的向量)，然后在特征后边加一个 1， 向量长度变成 513，记为 **a**

文件 [`target.txt`](classification/target.txt)  里有513个训练好的参数，这些参数组成一个向量，记为 **b**

计算 sigmoid(sum(**b** 点乘 **a**) ) 的值

如果大于 0.5，判断为男性

如果小于 0.5，判断为女性

##### [`./classification/train.py`](./classification/train.py) 		训练(Python版)

##### [`./classification/train.java`](classification/train.java) 	训练(Java版)

python 实在是太慢了，算到虎年都算不完👴

所以我把 train.py 移植到了 [train.java](classification/train.java), 点击 [这个bat脚本](classification/compile and run.bat) 编译并运行。每次运行训练的步数是写在train.java源码里的





### 人脸识别 recognition

##### [`./recognition/register.py`](./recognition/register.py) 注册人脸。

如果电脑有摄像头，可以直接运行该文件进行注册。

如果没有，也可以直接把图像文件放在文件夹`/recognition/account`里然后运行`register.py`。

##### [`./recognition/recognize.py`](./recognition/recognize.py) 通过摄像头识别人脸

通过摄像头实时获取图像，提取特征，与文件夹`account`里的`*.txt`中的特征相比对，找出最相似的人

.



```js
'有时用 CPU 运行会有这样的报错：';`
Traceback (most recent call last):
...
RuntimeError: [enforce fail at ..\c10\core\CPUAllocator.cpp:73] data. DefaultCPUAllocator: not enough memory: you tried to allocate 9437184 bytes. Buy new RAM!
...
`
'它竟然叫我买新内存条 ! ? ?'

```



