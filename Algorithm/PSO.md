# 粒子群算法

### 简介

粒子群算法（Particle Swarm Optimization），又称鸟群觅食算法，是由数学家J. Kennedy和R. C. Eberhart等开发出的一种新的进化算法。它是从随机解开始触发，通过迭代寻找出其中的最优解。本算法主要是通过适应度来评价解的分数，比传统的遗传算法更加的简单，它没有传统遗传算法中的“交叉”和“变异”等操作，它主要是追随当前搜索到的最优值来寻找到全局最优值。这种算法实现容易，精度高，收敛快等特点被广泛运用在各个问题中。
### 基本思想

粒子群算法是模拟鸟群觅食的所建立起来的一种智能算法，一开始所有的鸟都不知道食物在哪里，它们通过找到离食物最近的鸟的周围，再去寻找食物，这样不断的追踪，大量的鸟都堆积在食物附近这样找到食物的几率就大大增加了。粒子群就是这样一种模拟鸟群觅食的过程，粒子群把鸟看成一个个粒子，它们拥有两个属性——位置和速度，然后根据自己的这两个属性共享到整个集群中，其他粒子改变飞行方向去找到最近的区域，然后整个集群都聚集在最优解附近，最后最终找到最优解。

### 算法

算法中我们需要的数据结构，我们需要一个值来存储每个粒子搜索到的最优解，用一个值来存储整个群体在一次迭代中搜索到的最优解，这样我们的粒子速度和位置的更新公式如下：
 v[i] = w * v[i] + c1 * rand() * (pbest[i] - present[i]) + c2 * rand() * (gbest - present[i])    

          present[i] = present[i] + v[i] 
其中pbest是每个粒子搜索到的最优解，gbest是整个群体在一次迭代中搜索到的最优解，v[i]是代表第i个粒子的速度，w代表惯性系数是一个超参数，rang()表示的是在0到1的随机数。Present[i]代表第i个粒子当前的位置。我们通过上面的公式不停的迭代粒子群的状态，最终得到全局最优解。