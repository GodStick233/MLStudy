#吴恩达机器学习笔记（4）——正则化

> 这章我们主要探讨的是在机器学习中过拟合的情况如何处理

#### 过拟合

过拟合是机器学习中，计算机过度的为了达到目标，导致训练出的模型过度贴合训练集，导致我们使用其他数据就无法得出正确的结果。比如这样：

![](https://upload-images.jianshu.io/upload_images/8355793-af84daa8981bb981.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这个模型就过度贴合每一个数据了，导致加入新的数据，就无法得出正确的预测结果了。如何才能避免过拟合呢，如果我们的特征过多，我们可以删去几个无关紧要的特征，再进行训练。但是通常的情况，我们的每一次特征都是有用的，这里我们就需要来使用正则化这一方式

#### 代价函数

我们上面过拟合的图像是由下面这个模型拟合出来的：

![](https://upload-images.jianshu.io/upload_images/8355793-c6e6517fe1ac4cd3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可以看出，θ3和θ4这两个参数所在的项对函数图像平滑度影响最大，我们想要拟合出合适的图像，就必须给这两个参数惩罚，不能让他们过大，这样我们就要在代价函数后面加上1000(θ3^2)和1000(θ4^2)这样的话，我们的这两个参数如果过大，就会导致代价函数的代价值爆增。但是如果你的特征特别多，你就无法预知那个参数是在高阶项里，所以我们就需要把所以参数都约束住，所以我们的代价函数就变成了这样：

![](https://upload-images.jianshu.io/upload_images/8355793-298578ffbad108ec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

后面给每一个参数都约束住了，这个λ是正则参数，和学习率一样是我们自己设置，来控制参数的约束力度。
要注意的是我们只用约束θ1开始往后的参数

#### 线性回归正则化

线性回归我们知道，我们有两种方法来使我们的代价函数最小化，一种是梯度下降，一种是正规方程。这里我们就分别给大家两种不同的正则化方式。

- 梯度下降：
分为两个公式，一个是θ0的：

![](https://upload-images.jianshu.io/upload_images/8355793-1817dce32765e399.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

一个是θ1-θn：
![](https://upload-images.jianshu.io/upload_images/8355793-63283d9557e14da1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- 标准方程：
这里就直接给出公式：

![](https://upload-images.jianshu.io/upload_images/8355793-d30d2409fa5356ca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### Logistic 回归正则化

这里和回归很像也就直接给出公式了：

![](https://upload-images.jianshu.io/upload_images/8355793-75722cf5acd314fa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 后记
总算做的有惊无险的日更了，不过这一章内容不多，所以内容也不多，下一章我们将遇到重头戏神经网络，难度有点大，不知道明天能不能更新了。。