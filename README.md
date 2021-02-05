<h1 style="white-space:pre">rén  gōng  zhì  zhàng<br/>人      工      智      能</h1>

```python
""" 先看看作业的内容 """
#用feature tensor来计算欧式距离(可以用tensor，也可以转numpy)
#画1000个不同人特征的欧式距离的点和1000个相同人图片的特征的欧式距离
#附加题 用获取的feature 和 label来做分类（参考上一次的逻辑回归）
```

#### 23

##### [`./verification/showGraph.py`](./verification/showGraph.py) 显示欧氏距离图像

我把每张图片的特征算出来后直接保存到一些 .txt 文件里了 (相关代码在`writeFeature.py`文件里)。

虽然体积大了点，但是可以节省很多时间

#### 人脸识别 recognition

##### [`./recognition/register.py`](./recognition/register.py) 注册人脸。

如果电脑有摄像头，可以直接运行该文件进行注册。

如果没有，也可以直接把图像文件放在文件夹`/recognition/account`里然后运行`register.py`。

##### [`./recognition/recognize.py`](./recognition/recognize.py) 通过摄像头识别人脸

通过摄像头实时获取图像，提取特征，与文件夹`account`里的`*.txt`中的特征相比对，找出最相似的人

.



```js
'有时用 CPU 运行会有这样的报错：'
`
Traceback (most recent call last):
  File "d:/workspace/Projects_AI/Face/recognition/recognize.py", line 50, in <module>
    ft = p.getfeature(img)
  File "d:/workspace/Projects_AI/Face/recognition/recognize.py", line 31, in getfeature
    feature = self.arcface(F.interpolate(img, (112, 112), mode='bilinear', align_corners=True))
  File "D:\softs\Python\3.8.5\lib\site-packages\torch\nn\modules\module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "d:\workspace\Projects_AI\Face\recognition\network\resnet100.py", line 814, in forward
    stage4_unit2_conv2 = self.stage4_unit2_conv2(stage4_unit2_conv2_pad)
  File "D:\softs\Python\3.8.5\lib\site-packages\torch\nn\modules\module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "D:\softs\Python\3.8.5\lib\site-packages\torch\nn\modules\conv.py", line 423, in forward
    return self._conv_forward(input, self.weight)
  File "D:\softs\Python\3.8.5\lib\site-packages\torch\nn\modules\conv.py", line 419, in _conv_forward
    return F.conv2d(input, weight, self.bias, self.stride,
RuntimeError: [enforce fail at ..\c10\core\CPUAllocator.cpp:73] data. DefaultCPUAllocator: not enough memory: you tried to allocate 9437184 bytes. Buy new RAM!
[ WARN:0] global C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-kh7iq4w7\opencv\modules\videoio\src\cap_msmf.cpp (434) \`anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback
`
'它竟然叫我买新内存条 ! ? ?'

```


