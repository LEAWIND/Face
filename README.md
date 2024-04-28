***
> <h1 color="red">不要学我！</h1>
> 
> * 正式建立你的第一个Github项目前，建议先阅读[Github 文档](https://docs.github.com/zh/get-started/start-your-journey/about-github-and-git)
> * 不要在一个提交中包含 [1378 个修改](https://github.com/LEAWIND/Face/commit/33a1f92b634ef4d810e2c8eb15f717610b44c00f)
> * 提交最好遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 规范，不要写一堆[诡异的随机字符](https://github.com/LEAWIND/Face/commits/main/)。
> * 最好不要直接把数据集放在代码仓库里，如果需要分享数据集，可以上传到 [kaggle](https://www.kaggle.com) 上。
> * 注意代码风格保持统一，可以使用 [EditorConfig](https://editorconfig.org/) 来规范代码格式。
> * 最好使用 .gitattribute 来规范行分隔符，参考[模板](https://github.com/gitattributes/gitattributes)
> * 注意在 [.gitignore](https://docs.github.com/zh/get-started/getting-started-with-git/ignoring-files) 中忽略不必要的文件和目录，例如 `.vscode`, `.idea`
> * README 应当简洁明了，发布在github上的README最好[支持中英双语](https://github.com/LEAWIND/Third-Person/blob/1.20.1/README-ZH.md)。
> * 最好附上[开源协议](https://www.runoob.com/w3cnote/open-source-license.html)
***

任务

* 用feature tensor来计算欧式距离(可以用tensor，也可以转numpy)
* 画1000个不同人特征的欧式距离的点和1000个相同人图片的特征的欧式距离
* 附加题 用获取的feature 和 label来做分类（参考上一次的逻辑回归）

## 每个文件的作用

##### [`./writeFeature.py`](./writeFeature.py) 节省时间

把每张图片的特征算出来后直接保存到一些 .txt 文件里。

虽然体积有点大，但是可以加快速度。

### 作业 [0, 1]

* ##### [`./showGraph.py`](./showGraph.py) 显示欧氏距离图像

就是用 matplotlib 把图像画出来

顺便计算 "用来区分是否为同一人" 的阈值，如果欧氏距离大于这个阈值就可以认为不是同一人。



### 作业 [2] 性别分类 classification

我用两个函数模型分别试了试

模型1:

```python
# 训练集准确率
总: 0.938
男性: 0.927
女性: 0.893
# 测试集准确率
总: 0.805
男性: 0.671
女性: 0.934
```

模型2:

```python
# 训练集准确率
总: 0.889
男性: 0.898
女性: 0.88
# 测试集准确率
总: 0.859
男性: 0.780
女性: 0.934
```



#### 模型1 [文件夹在此](./classification)

先获取图像特征(长度为 512 的向量)，然后在后边加一个 1， 向量长度变成 513，记为 **a**

文件 [`target.txt`](classification/target.txt)  里有 513 个训练好的参数，这些参数组成一个向量，记为 **b**

计算 $sigmoid(sum(\pmb{a} · \pmb{b}) )$ 的值

如果大于 0.5，判断为男性

如果小于 0.5，判断为女性

* ##### [`./classification/train.py`](./classification/train.py) 		训练 (Python版)

* ##### [`./classification/train.java`](classification/train.java) 	训练 (Java版)

python 实在是太慢了，算到虎年都算不完👴 也许因为我的 GPU 是 MX350？

所以我把 train.py 移植到了 [train.java](classification/train.java), 用 [这个bat脚本](classification/compile and run.bat) 编译并运行。每次运行训练的步数是写在 train.java 源码里的。

每训练一定步数，就会把结果写在 [`target.txt`](classification/target.txt) 里。随时可以用它们做测试。

* ##### [`./classification/test.py`](./classification/test.py) 测试

用 `target.txt` 文件中的参数做测试，计算这堆参数在测试集的准确率。

* ##### [./classification/showGraph.py](./classification/showGraph.py) 显示图像

画出测试集中图片的性别预测值的分布图像

* ##### [`backup.py`](classification/backup.py), [`autoBackup.bat`](classification/autoBackup.bat) 备份`target.txt`文件

懒得 ctrl+C ctrl + V，所以用这个脚本每隔`60s`备份一次`target.txt`

就是复制到 [`backups`](classification/backup)文件夹里，然后文件名后边加个时间

* ##### [`./classification/prmChanging.py`](./classification/prmChanging.py) 参数变化趋势

根据 backup 中的数据绘制各参数的值随时间的变化的图像`

通过图像可以发现，很多参数的变化趋势近似抛物线或者双勾函数

* ##### [`compile and run.bat`](classification/compile and run.bat) 编译并运行`train.java`

只是因为懒得换 IDE

* ##### [`./classification/judge.py`](./classification/judge.py) 调用摄像头实时判断性别



#### 模型 2 [文件夹在此](./classification)

先获得特征：$ p = [x_1, x_2, ..., x_{513}] $

然后将 p 化成 ：$a = [p · p, p, 1] = [x_1^2, x_2^2, ..., x_{513}^2, x_1, x_2, ..., x_{513}, 1]$

此时 **a** 的长度是1025，target.txt 中的向量 **b**长度也是 1025

计算 $sigmoid(sum(\pmb{a} · \pmb{b}) )$ 的值

如果大于 0.5，判断为男性

如果小于 0.5，判断为女性

然后文件夹里的各个文件作用和 模型1 的相同。



### 人脸识别 recognition

* ##### [`./recognition/register.py`](./recognition/register.py) 注册人脸。

如果电脑有摄像头，可以直接运行该文件进行注册。

如果没有，也可以把 图像.jpg(.png...) 放在文件夹[`./recognition/account`](./recognition/account)里，然后直接运行`register.py`。

* ##### [`./recognition/recognize.py`](./recognition/recognize.py) 通过摄像头识别人脸

通过摄像头实时获取图像，提取特征，与文件夹`account`里的`*.txt`中的特征相比对，找出最相似的人

原理就是计算特征的欧氏距离。
