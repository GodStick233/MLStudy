# 吴恩达机器学习笔记（3）——Logistic 回归

> 放假这么久，天天摸鱼，已经好久没更新了，希望后面的更新速度能达到日更吧，这次给大家介绍的是Logistic 回归，虽然是名字带有回归，其实是一个分类算法。废话不多说，我们先从例题来引入我们今天的算法。

#### 引论
我们这次不讨论房价的问题了，这次我们来讨论肿瘤大小判断肿瘤是否是良性的肿瘤。这是一个两项分布问题，输出的结果只可能是两个一个是是另一个是否。我们可以用0,1来表示输出的结果。那么我们如何来区分良性还是恶性肿瘤呢，这就是一个典型的分类问题，我们也将通过本问题来学习Logistic 回归算法（虽然这个算法的名字含有回归，但是这不是一个回归问题而是分类）

#### 假设陈述
在Logistic 回归中我们希望函数的输出是在[0,1]这个范围。上一章我们提到我们的函数表达式表示成：

![](https://upload-images.jianshu.io/upload_images/8355793-42a32a81cea2dc69.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

但是我们在这里要把这个函数稍加修改变成h(x) = g((θ^T )*x)，而这个g(z) = 1/1+e^-z,这个就是大名鼎鼎的sigmoid函数，作用是使函数的输出在[0,1]这个范围。（在以后我们讲解神经网络部分中还会提到它），sigmoid函数的图像是这样的：

![](https://upload-images.jianshu.io/upload_images/8355793-d12caed1315cd914.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样我们就可以让函数的输出大于0.5的，表示为1，小于0.5的表示为0（反过来也可以），这样我们就可以不断的拟合参数θ，使函数的输出能达到这种分类的效果。

#### 决策界限

![](https://upload-images.jianshu.io/upload_images/8355793-7f13ccf06a04ad54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

假设我们现在有个训练集，就像上图所示。我们的假设函数是h(x)=g(θ0 + θ1*x1+θ2*x2)，假设我们已经拟合好了参数，参数的最终结果是[-3,1,1]。在sigmoid函数的图像中我们可以看出，当x>0时y>0.5,当x<0时y<0.5。所以当(θ^T )*x>=0时y=1,当(θ^T )*x<0时y=0。也就是-3+x1+x2>=0，化简我们可以得到x1+x2>3。
这在图像上是什么意思呢：

![](https://upload-images.jianshu.io/upload_images/8355793-2d57da2efbf57425.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可以看出我们拟合出来的函数已经把训练集划分开来了，x1+x2>3的区域就是函数的上半部分。到这里我们就明白了Logistic 回归的工作原理，就用拟合的函数来把不同标签的训练集分开来达到分类的效果。
对于不同的数据集分布，我们可以选择相应的函数图像来进行分割，比如这样的数据集分布，我们可以用圆形的函数来分割：

![](https://upload-images.jianshu.io/upload_images/8355793-ef43ded8ab880ae8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 代价函数

既然我们已经得到了算法的运作原理，那么我们又到了机器学习中最重要的环节，我们如何才能拟合。这里这个函数的代价函数又是什么？也就是优化的目标是什么？
在我们以前在线性回归中用到的代价函数是

![](https://upload-images.jianshu.io/upload_images/8355793-59bc1d714a7c4aa1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可不可以用这个函数来进行梯度下降呢，答案当然是不行的。因为我们在函数中使用了sigmoid函数，这会使我们代价函数的图像呈现这样的状态：

![](https://upload-images.jianshu.io/upload_images/8355793-292b8eab9a5de231.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样就会有很多个局部最小值，而达不到真正的代价的最小值。这里我们就引入了新的代价函数来对函数进行目标优化

![](https://upload-images.jianshu.io/upload_images/8355793-6f6ca58cdef00a56.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

把代价函数分为两种情况，一种是y=0一种是y=1的时候，这样我们可以分开来计算。这两个对数函数的图像，会使输出如果偏离1或者0的时候，代价值会爆增。

![y=1](https://upload-images.jianshu.io/upload_images/8355793-ad1bb026c278eae2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

但代价函数这样表示太麻烦了，于是我们可以把代价函数简化到一个公式里：

![](https://upload-images.jianshu.io/upload_images/8355793-cb75d672b8fe4c7f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样我们就不需要分情况来进行计算了

#### 梯度下降

我们既然得到了代价函数，我们的目标就是让代价函数最小化，我们就需要用梯度下降来得到这个目标。这里就和回归函数的梯度下降方法一样了，这里我就不多做介绍，直接给出梯度下降的公式：

![](https://upload-images.jianshu.io/upload_images/8355793-a54bd94d709ef4fb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/8355793-77e13a72fdf6b65a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 多元分类

上面我们提到的都是一分为二的情况，只需要分两种类别，但是在现实情况中我们可能需要处理的是分更多的类别，所以这个时候我们要怎么做呢。其实很简单，我们只需要把一个类别单独划分出来，与剩下的类别划分。这样一个一类别的划分，找到不用的函数，用多个函数来把各种类别区分出来，就可以了。

![](https://upload-images.jianshu.io/upload_images/8355793-5da9105d514535ae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 后记
终于恢复更新了，希望能达到日更吧，不能继续摸鱼了。。。。。
