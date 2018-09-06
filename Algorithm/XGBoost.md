# XGBoost算法

> 吴恩达的机器学习视频已经不能满足我了，断断续续又学了一些其他常见的机器学习算法，这里整理出来

### 决策树

决策树(Decision Tree）是在已知各种情况发生概率的上，通过构成决策树来求取净现值的)值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。由于这种决策分支画成图形很像一棵树的枝干，故称决策树。在机器学习中，决策树是一个预测模型，他代表的是对象属性与对象值之间的一种映射关系。

### XGboost算法

XGboost（eXtreme Gradient Boosting）算法是一种树的模型，常用于回归和分类。是一种梯度提升机器算法的扩展。原理是把大量的准确率较低的CART树通过组合形成一个准确率较高的模型。该模型每次训练迭代中都会生成新的树来减少误差。XGboost算法在每次迭代生成的树都会用梯度下降的方法，以上一个树为基础，向着最小化的目标来生成新的树。在一次一次迭代中生成大量的树来达到预期的期望。XGboost算法具有高准确，不容易过拟合等特点，在同类算法中脱颖而出。

#### CART树

CART树也叫回归树(regression tree)。CART树会把输入的属性分配到各个叶子节点，而且每个叶子节点上面会对应一个实数分数。从简单的类标到分数之后，我们可以做很多事情，如概率预测，排序。

#### Tree Ensemble

往往一个CART树太过于简单了，不能进行太复杂的情况，更不能有效的预测。因此我们需要把更多的CART树组合起来。我们可以通过把每一个CART树预测的结果分数，加在一起得到一个最终的分数，来当做预测的分数。这样可以减少误差，同时也能进行更加复杂情况下的预测。

#### 训练XGboost

**1.3.1目标函数**

Obj（θ） = L（θ） + Ω（θ）

上面是Xgboost的目标函数，是由误差函数和正则化项组成。常见的误差函数由平方误差，logistics误差函数等。正则化项是为了惩罚复杂模型，使模型不产生过拟合的现象。Xgboost的目标是使目标函数最小化。

**1.3.2数学模型**

![](https://upload-images.jianshu.io/upload_images/8355793-5a75d9c8e99df5a9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

XGBoost算法的核心就是每次生成一棵树都会更接近预期，这样一棵一棵树的生成来让整个模型的预测精度提高。我们可以用![](https://upload-images.jianshu.io/upload_images/8355793-14b7a1aaa5b88adf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 ，来表示模型的初始阶段，没有任何树。然后在![](https://upload-images.jianshu.io/upload_images/8355793-ea69850bd9a88f84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 生成第一颗树![](https://upload-images.jianshu.io/upload_images/8355793-84d8f8d4d6b361b7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 ，我们用![](https://upload-images.jianshu.io/upload_images/8355793-1d3e8eb30b3ee33e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 代表新的树的函数。就这样一棵一棵树的加入模型当中。所以最终生成第ｔ棵树时的模型表达式为：![](https://upload-images.jianshu.io/upload_images/8355793-28add65f1246de30.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 

最后我们要给函数加上惩罚项，来防止模型过拟合。这里在XGboost模型的惩罚项是：

![](https://upload-images.jianshu.io/upload_images/8355793-a08322ed9d907558.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中T是叶子的个数，ƴ是惩罚力度，是由我们定的，也就是叶子个数越多我们的惩罚力度也就越大。W代表的是每个叶子上面的分数，λ也是我们定的惩罚力度。

这样我们的模型最终的表达式为：

![](https://upload-images.jianshu.io/upload_images/8355793-79d2ee773e7f37ce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中constant是前t-1棵树的复杂度。我们再加上损失函数，表达式将会变成：

![](https://upload-images.jianshu.io/upload_images/8355793-9ea821774e1c4e6d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们再将非平方误差的表达式进行泰勒二阶展开：

![](https://upload-images.jianshu.io/upload_images/8355793-002503122cecdc3a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中gi和hi是

![](https://upload-images.jianshu.io/upload_images/8355793-5994499a3c03a6ea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 的一阶导数和二阶导数。具体表达式为：

![](https://upload-images.jianshu.io/upload_images/8355793-37f94af27b921ec8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然后把表达式中的常数项去掉，表达式就变成了：

![](https://upload-images.jianshu.io/upload_images/8355793-5e7838189b3f7158.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

就是最终的目标函数。这就是第t棵树的优化目标。我们在每次生成新的树都运用这个优化目标，使整个模型精确度更高。
