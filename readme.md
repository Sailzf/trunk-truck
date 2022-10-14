# #1 git
## （一）本地git操作
[主要参照：廖雪峰git教程][0]
[视频：黑马程序员Git全套教程][1]


1. 设置用户信息 【双引号之前加个空格】
```
$ git config --global user.name "用户名"
$ git config --global user.email "邮箱"
```
2. 查看配置信息
```
$ git config --global user.name
$ git config --global user.email
```
3. 基本操作类:
```
git init 初始化仓库
git-log 查看日志【非常重要】
git add 文件名.后缀 添加到暂存区
git commit -m "注释" 合并指定分支到当前活跃分支
```
4. 分支切换类
```
git checkout 分支名：切换到某个分支
git checkout -b 分支名：创建并切换到某个分支(创建一个新的分支并且直接使用)
```
5. 远程操作类
```
git clone 远程地址 本地文件夹：clone远程仓库到本地
git pull：拉取远端仓库的修改并合并
git push [--set-upstream] origin 分支名：推送本地修改到远端分支
（--set-upstream表示和远端分支绑定关联关系,只有第一次推送时才需要此参数）
```
6. 创建配置指令
```
$touch ~/.basrc打开文件创建 ：用于输出git提交日志
alias git-log='git log --pretty=oneline --all --graph --abbrev-commit'：用于输出当前目录所有文件及基本信息
alias ll='ls -al
```
7. 获取本地仓库
```
  git init 初始化当前目录作为一个git仓库
  ls -al 初始化完成后当前目录下会多一个.git文件夹
```

8. 基础操作指令
git工作目录下对于文件的增删改更新会存在几个状态,这些修改的状态会随着我们执行git的命令而发生变化。
```
git add (工作区 --> 暂存区) . 为通配符 add所有的暂存区
git commit -m "标题名称,方便查看"
git status  查看修改的状态 status(暂存区、工作区)
```
9. 进入vi编辑器
```
  vi 文件名.后缀：按键盘上的[Insert]键进入编辑,[ins]
```
10. 合并快进模式 快捷合并修改内容
```
$git checkout 主分支名
$git merge 分支名
```

##（二）github

1. git远程仓库:常用的有git hub、码云、gitlab等
2. 配置SSH公钥
```
ssh-keygen -t rsa 生成SSH公钥, 不断回车
cat ~/.ssh/id_rsa.pub 获取公钥
ssh -T git@gitee.com 查看公钥是否连接成功
mv ~/.ssh ~/.ssh.bark2 备份公钥
```

3. 绑定远程仓库
```
1.绑定远程仓库,在gitee中找到自己想要绑定的远程仓库,然后找到ssh/https复制链接
2.$git remote add origin(默认是origin/仓库名) 远程仓库路径
3.$git remote 获取仓库名称是否配置成功 
4.$git push [-f] [--set-upstream] [远程名称[本地分支名][:远程分支名]] 代码推送到远程仓库
```

[秘钥设置][2]

# #2 Markdown Test
## title2
para1
*斜体文本*
_斜体文本_
**粗体文本**
**粗斜体文本**
***
---
~~删除线~~
<u>下划线</U>
创建脚注样例[okok]
[^okok]:一个一个脚注

* *+空格 无序列表

1.第一项 有序列表
  * 两个空格
    * 四个空格
    >　四个空格才能区块
      * 还有
        * 还有

> 区块引用
> 继续
>>还可以嵌套

`printf()`这是一个函数

    dataset = datasets.load_iris()

[链接名称][https://d.jotang.club/t/topic/804/4]
还可以为链接赋以变量名: [jotang][3]

![文字替代图片](图片地址)
![文字替代图片](http://static.runoob.com/images/runoob-logo.png "图片地址标题")
<img src="http://static.runoob.com/images/runoob-logo.png" width="50">


# #3 Linux&WSL
##一、Linux
### 1. 什么是linux？
Linux是一套免费使用和自由传播的类Unix操作系统，是一个多用户、多任务、支持多线程和多CPU的操作系统。
### 2. Linux系统的特点？
* 稳定的系统 ：打个比方，安装Linux的主机连续运行一年以上不曾宕机、不必关机是很平常的事，我的windows系统今早打开时是黑屏，强行关机之后才恢复，原因不明
* 安全性和漏洞的快速修补 ：Linux有众多电脑高手在使用，所以维护者众多，更新维护很快，而windows则是所有人都会用，且不开源
* 多任务，多用户 ：你可以在一个Linux主机上规划出不同等级的用户，而且每个用户登录系统时工作环境可以不同，此外你还可以允许不同用户在同一时间登陆主机以使用主机的资源
* 相对较少的系统资源占用 ：这是最吸引眼球的地方，目前市面上任何一款个人计算机都可以达到使用Linux搭建一个服务上百人以上的主机
* 可定制裁剪：移植到嵌入式平台（如安卓设备） 
* 可选择的多种图形用户界面（如GNOME,KDE）
* linux没有盘符这个概念，只有一个根目录 /，所有文件都在它下面
Centos的文件结构
> /bin 可执行二进制文件的目录，如常用的命令 ls、tar、mv、cat 等
/home 普通用户的家目录
/root root用户的家目录
/boot 内核文件的引导目录, 放置 linux 系统启动时用到的一些文件
/sbing 超级用户使用的指令文件
/tmp 临时文件目录,一般用户或正在执行的程序临时存放文件的目录，任何人都可以访问，重要数据不可放置在此目录下。
/dev 设备文件目录 万物皆文件
/lib 共享库，系统使用的函数库的目录，程序在执行过程中，需要调用
### 3. win系统的特点
直观、高效的面向对象的图形用户界面，易学易用，Windows用户界面和开发环境都是面向对象的，这种操作方式模拟了现实世界的行为，易于理解、学习和使用。

## 二、WSL
### 1. What?
适用于 Linux 的 Windows 子系统(简称WSL-Windows Subsystem for Linux),可让开发人员直接在 Windows 上按原样运行 GNU/Linux 环境.
### 2. How?
[WSL(Linux 子系统)安装实践][4]
[WIN11安装实践][5]
### 3.problem
* *`wsl --install 正在安装: 虚拟机平台 已安装 虚拟机平台。 正在安装: 适用于 Linux 的 Windows 子系统 已安装 适用于 Linux 的 Windows 子系统。 正在下载: WSL 内核 安装过程中遇到错误，但可以继续安装。组件： ‘WSL 内核’ 错误代码： 0x80072f78`*
  勾选“Windows虚拟机监控程序平台”。然后卸载WSL（勾除“适用于Linux的Windows子系统”）、重启、再重装WSL（勾选“适用于Linux的Windows子系统”/输入命令wsl --install）即可。
* *`The Windows Subsystem for Linux optional component is not enabled. Please enable`*
控制面板→程序→启用或关闭Windows功能，在列表中找到适用于Linux的Windows子系统，勾选上，点击确定.
* *`正在下载: Ubuntu-20.04,安装过程中出现错误。分发名称: 'Ubuntu-20.04' 错误代码: 0x80072f7d`*
直接在Microso Store里安装Ubuntu：
* *`0x80004005`*
  * 在命令提示符窗口中输入字符串“regsvr32 Softpub.dll”并回车键确定，之后弹出“DllRegisterServer在Softpub.dll已成功”对话框，点击确定。
  * 再输入字符串“regsvr32 Wintrust.dll”并回车键确定，弹出“DllRegisterServer在Wintrust.dll已成功”对话框，
  * 最后再输入“regsvr32 Initpki.dll”回车键确定，弹出DllRegisterServer在regsvr32 initpki.dll已成功的窗口
* *`请启用虚拟机平台 Windows 功能并确保在 BIOS 中启用虚拟化。`*
[解决方法][6]



# #8 人工智能


# 一、机器学习理论
## （零) 重点关注
### 1. 监督学习&无监督学习
#### 1.1 监督学习（supervised learning）
从给定的训练数据集中学习出一个函数（模型参数），当新的数据到来时，可以根据这个函数预测结果。监督学习的训练集要求包括输入输出，通过已有的训练样本（即已知数据及其对应的输出）去训练得到一个最优模型，再利用这个模型将所有的输入映射为相应的输出，对输出进行简单的判断从而实现分类的目的。也就具有了对未知数据分类的能力。监督学习的目标往往是让计算机去学习我们已经创建好的分类系统（模型）。

监督学习是训练神经网络和决策树的常见技术。这两种技术高度依赖事先确定的分类系统给出的信息，对于神经网络，分类系统利用信息判断网络的错误，然后不断调整网络参数。对于决策树，分类系统用它来判断哪些属性提供了最多的信息。

> 有监督学习最常见的就是：regression&classification
  * Regression：Y是实数vector。回归问题，就是拟合(x,y)的一条曲线，使得价值函数(costfunction) L最小
  * Classification：Y是一个有穷数(finitenumber)，可以看做类标号，分类问题首先要给定有lable的数据训练分类器，故属于有监督学习过程。分类过程中cost function l(X,Y)是X属于类Y的概率的负对数。

#### 1.2 无监督学习（unsupervised learning）
输入数据没有被标记，也没有确定的结果。样本数据类别未知，需要根据样本间的相似性对样本集进行分类（聚类，clustering）试图使类内差距最小化，类间差距最大化。通俗点将就是实际应用中，不少情况下无法预先知道样本的标签，也就是说没有训练样本对应的类别，因而只能从原先没有样本标签的样本集开始学习分类器设计。

> 无监督学习的方法分为两大类：
* 一类为基于概率密度函数估计的直接方法：指设法找到各类别在特征空间的分布参数，再进行分类。
* 另一类是称为基于样本间相似性度量的简洁聚类方法：其原理是设法定出不同类别的核心或初始内核，然后依据样本与核心之间的相似性度量将样本聚集成不同的类别。
利用聚类结果，可以提取数据集中隐藏信息，对未来数据进行分类和预测。应用于数据挖掘，模式识别，图像处理等。

### 2. 分类器和预测器
#### 2.1 预测器
将计算机看成一个机器，输入一个数得到输出（假设函数是y=kx最简单的线性函数）。那么我们如何使用这个函数由输入预测输出呢？答案要从过去中来。如果我们知道一组数据并且想要发现这组输入输出数据的关系从而达到预测的目的那么我们需要不断地比较已有输出和预测输出之间的误差，并且设计一个合理地调节梯度。通过多次训练最终拟合出最合适的函数（即误差最小）。
#### 2.2 梯度
调整的幅度即为梯度。如预测的输出大于实际输出，那么我们可以调整k使其增加0.1，调整的幅度即为梯度。调整幅度的过程涉及到机器学习中至关重要的概念学习率：ΔK=L(E/x)。这里E/x即(y-y’)/k只是对于上文提到的函数y=kx而言，即每次调整K的幅度应该是不断的迭代（E为误差值即E=y-y’预测值与实际值之间的误差，L标记为学习率由我们设置如L=0.5每次只学习预期调整幅度的一半）。这样做的好处是避免训练数据本身的不确定性所带来的误差，可以有节制地抑制某些极端样本，更有利于还原整体情况。
##### 2.2.1 数学意义
在微积分里面，对多元函数的参数求偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度。

##### 2.2.2 几何意义：
函数变化增加最快的地方。具体来说，对于函数,在点，沿着梯度向量的方向就是的方向是增加最快的地方。或者说，沿着梯度向量的方向，更加容易找到函数的最大值。反过来说，沿着梯度向量相反的方向，也就是 的方向，梯度减少最快，也就是更加容易找到函数的最小值。
##### 2.2.3 个人理解
梯度指向值变化最大的方向！所以负梯度就是指向最优权重。

#### 2.3 分类器
简单的线性函数能帮我们起到分类的作用。显然一条直线将整个坐标平面分为了上下两个部分，即分为了两类。一个有趣的例子可以证明这个极简单的分类器的功效。
花园里有毛毛虫和瓢虫，毛毛虫很长而瓢虫却很宽，如果我们把长和宽作为x与Y放入坐标轴中就可以通过该坐标在直线的上部或下部轻松地区别出毛毛虫与瓢虫即分类的功能得到实现（然后很快我们就发现该分类器是线性的无法实现一些相对复杂的功能。如：XOR逻辑功能，此时我们需要多个分类器辅助我们完成这个功能）。

### 3. 独热码
#### 3.1 what
独热编码（One-Hot Encoding），又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，其中只有一位有效。即，只有一位是1，其余都是零值。

> 例如，对六个状态进行编码：
自然顺序码为 000,001,010,011,100,101
独热编码则是 000001,000010,000100,001000,010000,100000

#### 3.2 why

在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的。而常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间。
使用独热编码（One-Hot Encoding），将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。将离散型特征使用独热编码（One-Hot Encoding），会让特征之间的距离计算更加合理。

### 4. 向量化
#### 4.1 what
将数据以向量的形式输入。
#### 4.2 why
一言蔽之，**向量化快！**
向量化可以去除代码中 的for 循环。当在深度学习领域，代码中显式地使用 for 循环使算法很低效，如果在大数据集上，代码可能花费很长时间去运行，
向量化的实现将会非常直接计算 wT*x 。使用两个方法——向量化和非向量化，计算相同的值，其中向量化版本花费了0.968毫秒，而非向量化版本的 for 循环花费了327.997毫秒，大概是300多倍。
### 5.学习率
#### 5.1 梯度
见2.2
#### 5.2 学习率
“下山步子”的大小。
#### 5.3 损失函数(Loss Function 或 Cost Function)
代价函数也称为损失函数。代价函数并没有准确的定义，可以理解为是一个人为定义的函数，我们可以利用这个函数来优化模型的参数。最简单且常见的一个代价函数是均方差(MSE)代价函数，也称为二次代价函数。（就是方差）
#### 5.4 过拟合
[过拟合][18] ：简单来说，就是训练的函数不适用于测试集

### 6. 神经元
神经元模型模拟大脑神经元的运行过程，其包含输入，输出与计算功能，输入可以类比为神经元的树突，而输出可以类比为神经元的轴突，计算则可以类比为细胞核。下图是一个典型的神经元模型：包含有m个输入，1个输出，以及2个计算功能。
![神经元](https://s3.bmp.ovh/imgs/2022/10/14/fcac82b398f61973.jpg)
## （一）神经网络
### 0. 分类
神经网络可以分为三种主要类型：前馈神经网络、反馈神经网络和图神经网络。

### 1. 前馈神经网络
前馈神经网络（feedforward neural network）是一种简单的神经网络，也被称为多层感知机（multi-layer perceptron，简称MLP），其中不同的神经元属于不同的层，由输入层-隐藏层-输出层构成，信号从输入层往输出层单向传递，中间无反馈。

图1 前馈神经网络结构图
![文字替代图片](图片地址)
#### 1.1 组成
前馈神经网络中包含激活函数（sigmoid函数、tanh函数等）、损失函数（均方差损失函数、交叉熵损失函数等）、优化算法（BP算法）等。
#### 1.2 例子
常用的模型结构有：卷积神经网络、BP神经网络、RBF神经网络、感知器网络等。
#### 1.3 实例：卷积神经网络 CNN
[卷积神经网络][12]（Convolutional Neural Networks, CNN）是一类包含卷积运算且具有深度结构的前馈神经网络（Feedforward Neural Networks）。
##### 1.3.0 what
整体架构：输入层—卷积层—池化层—全连接层—输出层

##### 1.3.1 输入层
以图片为例，输入的是一个三维像素矩阵，长和宽表示图像的像素大小，深度表示色彩通道（黑白为1，RGB彩色为3）。

##### 1.3.2 卷积层
卷积层也是一个三维矩阵，它的每个节点（单位节点矩阵）都是上一层的一小块节点（子节点矩阵）加权得来，一小块的尺寸一般取3*3或5*5。此层的作用是对每一小快节点进行深入分析，从而提取图片更高的特征。

##### 1.3.3 池化层
池化层不会改变三维矩阵的深度，其作用是缩小矩阵，从而减少网络的参数。

##### 1.3.4 全连接层
跟**全连接神经网络**作用一样。
##### 1.3.5 激活层
负责对卷积层抽取的特诊进行激活，由于卷积操作是把输入图像和卷积核进行相应的线性变换，需要引入激活层(非线性函数)对其进行非线性映射。激活层由非线性函数组成，常见的如sigmoid、tanh、relu。最常用的激活函数是Relu，又叫线性整流器。
  * **why?**  线性函数没有上界，经常会造成一个节点处的数字变得很大很大，难以计算，也就无法得到一个可以用的网络。因此人们后来对节点上的数据进行了一个操作，如利用sigmoid()函数来处理，使数据被限定在一定范围内。

* Softmax层：得到当前样例属于不同种类的概率分布，并完成分类。

##### 1.3.6 why
相比早期的BP神经网络，卷积神经网络最重要的特性在于“参数共享”与“局部感知”。
* 权重共享
传统的神经网络的参数量巨大，***全连接神经网络*** 最大的问题就是权值参数太多，而卷积神经网络的卷积层，不同神经元的权值是共享的，这使得整个神经网络的参数大大减小，提高了整个网络的训练性能。
* 局部感知
一张图像，我们实际上并不需要让每个神经元都接受整个图片的信息，而是让不同区域的神经元对应一整张图片的不同局部，最后只要再把局部信息整合到一起就可以了。这样就相当于在神经元最初的输入层实现了一次降维
* 平移不变性
即使图片中的目标位置平移到图片另一个地方，卷积神经网络仍然能很好地识别出这个目标，输出的结果也和原来未平移之前是一致的。
##### 1.3.7 卷积
[卷积的本质：先将一个函数翻转，然后进行滑动叠加。][13]
#### 1.4 BP神经网络
[Bp神经网络][11]可以**分为两个部分**，bp和神经网络。bp是 Back Propagation 的简写，意思是反向传播。
> 一个通俗的例子，猜数字：
提前设定一个数值 50，通过猜的数字是高了还是低了得到正确答案。


### 2. 反馈神经网络
反馈神经网络（feedback neural network）的输出不仅与当前输入以及网络权重有关，还和网络之前的输入有关。常用的模型结构有：RNN、Hopfield网络、玻尔兹曼机、LSTM等。

![文字替代图片](图片地址)
图3 反馈神经网络结构图

#### 2.1 实例：循环神经网络 RNN：
> 在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的。但是这种普通的神经网络对于很多问题都无能无力。比如你要预测句子的下一个单词是什么，一般需要用到前面的单词。

RNN之所以称为循环神经网络，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即**隐藏层之间的节点不再无连接而是有连接的**，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。
![RNN神经网络的结构](图片地址)

### 3. 图神经网络
图（graph）是一种在拓扑空间内按图结构组织来关系推理的函数集合，包括社交网络、知识图谱、分子图神经网络等。
GNN是直接在图数据结构上运行的神经网络。GNN的典型应用便是节点分类。
相比较于神经网络最基本的网络结构全连接层（MLP），特征矩阵乘以权重矩阵，图神经网络多了一个邻接矩阵。计算形式很简单，三个矩阵相乘再加上一个非线性变换：

## （二） 如果能重来，我会按这个顺序学习：
[1.先用最快时间看吴恩达讲理论,][14]
[2.卷积神经网络图像化理解(从52p开始),][15]
[3.零基础，理论+实操的帖子][16]+[4.莫烦，实操的视频][17]
[5.再看李沐的巩固一遍理论和实操][17]
## 二、TASK1
## (一) 环境配置
### 1. What is anaconda?
Anaconda是一个用于科学计算的Python发行版，支持 Linux, Mac, Windows系统，提供了包管理与环境管理的功能，可以很方便地解决多版本python并存、切换以及各种第三方包安装问题。

* conda可以理解为一个工具，也是一个可执行命令，其核心功能是包管理与环境管理。包管理与pip的使用类似，环境管理则允许用户方便地安装不同版本的python并可以快速切换。
* Anaconda则是一个打包的集合，里面预装好了conda、某个版本的python、众多packages、科学计算工具等等，所以也称为Python的一种发行版。

### 2. Conda的环境管理
Conda的环境管理功能允许我们同时安装若干不同版本的Python，并能自由切换。对于上述安装过程，假设我们采用的是Python 2.7对应的安装包，那么Python 2.7就是默认的环境（默认名字是root，注意这个root不是超级管理员的意思）。

假设我们需要安装Python 3.4，此时，我们需要做的操作如下：
```# 创建一个名为python34的环境，指定Python版本是3.4（不用管是3.4.x，conda会为我们自动寻找3.4.x中的最新版本）
conda create --name python34 python=3.4

# 此时，再次输入
python --version
# 可以得到`Python 3.4.5 :: Anaconda 4.1.1 (64-bit)`，即系统已经切换到了3.4的环境

# 如果想返回默认的python 2.7环境，运行
deactivate python34 # for Windows
source deactivate python34 # for Linux & Mac

# 删除一个已有的环境
conda remove --name python34 --all

# 安装好后，使用activate激活某个环境
activate python34 # for Windows
source activate python34 # for Linux & Mac
# 激活后，会发现terminal输入的地方多了python34的字样，实际上，此时系统做的事情就是把默认2.7环境从PATH中去除，再把3.4对应的命令加入PATH 
```

用户安装的不同python环境都会被放在目录~/anaconda/envs下，可以在命令中运行conda info -e查看已安装的环境，当前被激活的环境会显示有一个星号或者括号。
### 3. Conda的包管理
Conda的包管理就比较好理解了，这部分功能与pip类似。

#### 3.1 例:安装scipy
```
# 安装scipy
conda install scipy
# conda会从从远程搜索scipy的相关信息和依赖项目，对于python 3.4，conda会同时安装numpy和mkl（运算加速的库）

# 查看已经安装的packages
conda list
# 最新版的conda是从site-packages文件夹中搜索已经安装的包，不依赖于pip，因此可以显示出通过各种方式安装的包
```
#### 3.2 conda的一些常用操作
```
# 查看当前环境下已安装的包
conda list

# 查看某个指定环境的已安装包
conda list -n python34

# 查找package信息
conda search numpy

# 安装package
conda install -n python34 numpy
# 如果不用-n指定环境名称，则被安装在当前活跃环境
# 也可以通过-c指定通过某个channel安装

# 更新package
conda update -n python34 numpy

# 删除package
conda remove -n python34 numpy
```
前面已经提到，conda将conda、python等都视为package，因此，完全可以：
#### 3.3使用conda来管理conda和python的版本

```
# 更新conda，保持conda最新
conda update conda

# 更新anaconda
conda update anaconda

# 更新python
conda update python
# 假设当前环境是python 3.4, conda会将python升级为3.4.x系列的当前最新版本

补充：如果创建新的python环境，比如3.4，运行conda create -n python34 python=3.4之后，conda仅安装python 3.4相关的必须项，如python, pip等，如果希望该环境像默认环境那样，安装anaconda集合包，只需要：

# 在当前环境下安装anaconda包集合
conda install anaconda

# 结合创建环境的命令，以上操作可以合并为
conda create -n python34 python=3.4 anaconda
# 也可以不用全部安装，根据需求安装自己需要的package即可
```

#### 3.4 下载速度
如果需要安装很多packages，你会发现conda下载的速度经常很慢，因为Anaconda.org的服务器在国外。所幸的是，清华TUNA镜像源有Anaconda仓库的镜像，我们将其加入conda的配置即可：
```
# 添加Anaconda的TUNA镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# TUNA的help中镜像地址加有引号，需要去掉

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
```

执行完上述命令后，会生成~/.condarc(Linux/Mac)或C:\Users\USER_NAME\.condarc文件，记录着我们对conda的配置，直接手动创建、编辑该文件是相同的效果。


## （二) 各个库的作用
### 0. Iris
Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa(山鸢尾)，Versicolour(杂色鸢尾)，Virginica(维吉尼亚鸢尾)）三个种类中的哪一类。

### 1. numpy
#### 1.1 基本属性
NumPy的主要对象是同构多维数组。它是一个元素（通常是数字）表，所有元素（通常是数字）都具有相同的类型，由非负整数元组索引。在NumPy中，维度称为轴。
数组对象的基本属性：
`ndarray.ndim`	数组的轴数（维数）
`ndarray.shape`	数组的尺寸。返回一个整数元组，元组的长度是轴数，元组中的元素指示每个维度中数组的大小。
`ndarray.size`	数组的元素总数。这等于shape元组中所有元素的乘积。
`ndarray.dtype`	描述数组中元素类型的对象。可以使用标准Python类型创建或指定dtype。此外，NumPy还提供了自己的类型，如numpy.int32、numpy.int16 、numpy.float64
`ndarray.itemsize`	数组中每个元素的大小（以字节为单位）

#### 1.2 创建数组
1. 从常规Python列表或元组出发创建数组
`a=np.array([1,2,3,4,5],dtype=float)`  # 列表转换为一维数组
`b=np.array([(1,2,3,4),(5,6,7,8)])  `  # 元组的列表转换为二维数组
`c=np.array([[( 1, 2, 3, 4),( 5, 6, 7, 8)],[( 9,10,11,12),(13,14,15,16)],[(17,18,19,20),(21,22,23,24)]])`
2. 创建具有初始占位符内容的数组
`a=np.zeros((2,3))`  # 创建一个充满0的数组,参数是表明数组形状的元组
`b=np.ones((2,3))`     # 创建一个充满1的数组,参数是表明数组形状的元组
`c=np.empty((2,3,4))`  # 创建一个数组,其初始内容是随机的
3. 创建数字序列
`np.arange(x,y,z)` 创建一个 [ x , y ) [x,y)[x,y) 内以z为步长的数组
`np.linspace(x,y,z)` 创建一个[ x , y ] [x,y][x,y] 内步长相同有z个值的数组（平均分成z-1份）

plt.plot():matplotlib.pyplot用于画图的函数
plt.show():将plt.xx()处理后的函数显示出来
### 2. sklearn
[sklearn简介][7]
主要子模块：样例数据集、数据预处理、特征选择、模型选择、度量指标、降维、聚类、基本学习模型、集成学习模型。
![sklearn](https://s3.bmp.ovh/imgs/2022/10/14/077b22dad612138f.png)
### 3. pytorch
#### 3.1 torch
torch:张量的相关运算，eg:创建、索引、切片、连续、转置、加减乘除等相关运算。
torch.nn:包含搭建网络层的模块（modules)和一系列的loss函数。eg.全连接、卷积、池化、BN分批处理、dropout、CrossEntropyLoss、MSLoss等损失函数。
torch.autograd:提供Tensor所有操作的自动求导方法。
torch.nn.functional:常用的激活函数relu、leaky_relu、sigmoid等。
torch.optim:各种参数优化方法，例如SGD、AdaGrad、RMSProp、Adam等。
torch.nn.init:可以用它更改nn.Module的默认参数初始化方式。
torch.utils.data:用于加载数据。
torch.Tensor模块定义了torch中的张量类型，

## （三）其它问题
[Google 人机验证的国内加载，包括reCaptcha 验证码][8]

# 三、TASK2-Field 1-Computer Vision
[toc]
## （零）小问题
#### 1. 计算机以怎样的方式存储一张图片，又是怎样显示它的？
##### 1.1 黑白或灰度图像
举个例子。这里采取了黑白图像，也被称为一个灰度图像
![灰度图像](https://s3.bmp.ovh/imgs/2022/10/14/d86f641d3857e4d3.png)
这些小方框叫做Pixels（像素）。我们经常使用的图像维度是X x Y（X by Y）。这些像素中的每一个都表示为数值，而这些数字称为像素值。这些像素值表示像素的强度。对于灰度或黑白图像，我们的像素值范围是0到255。接近零的较小数字表示较深的阴影，而接近255的较大数字表示较浅或白色的阴影。

因此，计算机中的每个图像都以这种形式保存，其中你具有一个数字矩阵，该矩阵也称为Channel（通道）

##### 1.2 彩色图像
![彩色图像](https://s3.bmp.ovh/imgs/2022/10/14/b2c256c4f736c671.png)

该图像由许多颜色组成，几乎所有颜色都可以从三种原色（红色，绿色和蓝色）生成。我们可以说每个彩色图像都是由这三种颜色或3个通道（红色，绿色和蓝色）。红绿蓝三个通道和每个通道具体像素都具有从0到255的值，其中每个数字代表像素的强度，或者可以说红色，绿色和蓝色的阴影。最后，所有这些通道或所有这些矩阵都将叠加在一起。

#### 2. 作为一名 programmer，我们在代码中采用什么数据结构将图片读入？
数组结构。

#### 3. 如今CV领域已然发展得非常成熟，有哪些研究领域（目标检测、图像生成……）？

##### 3.1 图像分类
图像分类是计算机视觉中重要的基础问题，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层视觉任务的基础。

对于一幅图像来说，分类就是聚类，即分割；对于一组图像样本来说，分类是检测出样本中有相同目标的图像。

##### 3.2 目标检测
目标检测任务是给定一张图像或是一个视频帧，让计算机定位出这个目标的的位置并且知道目标物是什么，即输出目标的Bounding Box（边框）以及分类标签。

##### 3.3 目标分割
目标分割是检测到图像中的所有目标，分为语义分割（Semantic-level）和实例分割（Instance-level），
###### 3.3.1 语义分割
图像语义是指对图像内容的理解，例如，能够描绘出什么物体在哪里做了什么事情等，分割是指对图片中的每个像素点进行标注，标注属于哪一类别。通常意义上的目标分割指的就是语义分割。
###### 3.3.2 实例分割
实例分割其实就是目标检测和语义分割的结合。相对语义分割，实例分割需要标注出图上同一物体的不同个体（羊1，羊2，羊3…）。
##### 3.4 目标跟踪
目标跟踪是指在给定场景中跟踪感兴趣的具体对象或多个对象的过程。

##### 3.5 总结
计算机视觉典型的技术路线是：目标分割 ——>目标检测 ——>目标识别 ——>目标跟踪

> 如：需要对视频中的小明进行跟踪，处理过程将经历如下过程：
（1）首先，采集第一帧视频图像，因为人脸部的肤色偏黄，因此可以通过颜色特征将人脸与背景分割出来（目标分割）；
（2）分割出来后的图像有可能不仅仅包含人脸，可能还有部分环境中颜色也偏黄的物体，此时可以通过一定的形状特征将图像中所有的人脸准确找出来，确定其位 置及范围（目标检测）；
（3）接下来需将图像中的所有人脸与小明的人脸特征进行对比，找到匹配度最好的，从而确定哪个是小明（目标识别）；
（4）之后的每一帧就不需要像第一帧那样在全图中对小明进行检测，而是可以根据小明的运动轨迹建立运动模型，通过模型对下一帧小明的位置进行预测，从而提升跟踪的效率（目标跟踪）

#### 4. 了解一个非常重要的库OpenCV
见（五）
## （一）环境配置
### 1. **显卡和GPU**
**显卡**（Video card，Graphics card）是显示卡的简称，由GPU、显存等组成，主机里的数据要显示在屏幕上就需要显卡。因此，显卡是电脑进行数模信号转换的设备，承担输出显示图形的任务，将电脑的数字信号转换成模拟信号让显示器显示出来。原始的显卡一般都是集成在主板上，只完成最基本的信号输出工作，并不用来处理数据。随着显卡的迅速发展，就出现了GPU的概念。
**GPU**（Graphic Processing Unit，图形处理单元）是图形处理器，一般焊接在显卡上的。GPU是显卡上的一块芯片，就像CPU是主板上的一块芯片。大部分情况下，我们所说GPU就等同于指显卡，但是实际情况是GPU是显示卡的“心脏”，是显卡的一个核心零部件，核心组成部分。
* **GPU的由来与发展**
GPU由 NVIDIA 公司于1999年提出的。之前显卡也有GPU，只不过没有命名。
仅用于图形渲染，此功能是GPU的初衷，后来人们发现，GPU这么一个强大的器件只用于图形处理太浪费了，它应该用来做更多的工作，例如浮点运算。怎么做呢？直接把浮点运算交给GPU是做不到的，因为它只能用于图形处理（那个时候）。最容易想到的，是把浮点运算做一些处理，包装成图形渲染任务，然后交给GPU来做。这就是 GPGPU（General Purpose GPU）的概念。
于是，为了让不懂图形学知识的人也能体验到GPU运算的强大，NVIDIA公司又提出了CUDA的概念。

### 2. **CUDA**
CUDA(Compute Unified Device Architecture)，通用并行计算架构，是一种运算平台。
它包含 CUDA 指令集架构以及 GPU 内部的并行计算引擎。你只要使用一种类似于C语言的 CUDA C 语言，就可以开发CUDA程序，从而可以更加方便的利用GPU强大的计算能力，而不是像以前那样先将计算任务包装成图形渲染任务，再交由GPU处理。
    > 简单来讲，比如通过CUDA架构，视频播放软件可以充分挖掘NVIDIA系列显卡的GPU并行计算能力，轻松进行高清影片的播放，与软件高清解码相比，CPU占用可以下降一半以上。当然，CUDA的应用领域绝不仅仅是视频、图形、游戏，包括各种3D和建模，医疗、能源、科学研究等，到处都可见到这种技术架构的应用。

### 3. **cuDNN**
cuDNN是CUDA在深度学习方面的应用。使得CUDA能够应用于加速深度神经网络。
NVIDIA cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销,可以集成到更高级别的机器学习框架中，如谷歌的Tensorflow。简单的插入式设计可以让开发人员专注于设计和实现神经网络模型。
    > cuDNN是基于CUDA的深度学习GPU加速库，有了它才能在GPU上完成深度学习的计算。它就相当于工作的工具，比如它就是个扳手。但是CUDA这个工作台买来的时候，并没有送扳手。想要在CUDA上运行深度神经网络，就要安装cuDNN，就像你想要拧个螺帽就要把扳手买回来。这样才能使GPU进行深度神经网络的工作，工作速度相较CPU快很多。

    > cuDNN是CUDA的扩展计算库。
只要把cuDNN文件复制到CUDA的对应文件夹里就可以，即是所谓插入式设计。CUDA已有的文件与cuDNN没有相同的文件，复制cuDNN的文件后，CUDA里的文件并不会被覆盖，所以CUDA中的其他文件并不会受影响。

### 4. **NVIDIA**
NVIDIA（英伟达）是一家人工智能计算公司。

### 5. 其他
#### 环境变量是什么？
> 环境变量一般是指在操作系统中用来指定操作系统运行环境的一些参数，如：临时文件夹位置和系统文件夹位置等。环境变量是在操作系统中一个具有特定名字的对象，它包含了一个或者多个应用程序所将使用到的信息。例如Windows和DOS操作系统中的path环境变量，当要求系统运行一个程序而没有告诉它程序所在的完整路径时，系统除了在当前目录下面寻找此程序外，还应到path中指定的路径去找。用户通过设置环境变量，来更好的运行进程。
#### dll文件是什么？
  > DLL(Dynamic Link Library)文件为动态链接库文件，又称“应用程序拓展”，是软件文件类型。在Windows中，许多应用程序并不是一个完整的可执行文件，它们被分割成一些相对独立的动态链接库，即DLL文件，放置于系统中。当我们执行某一个程序时，相应的DLL文件就会被调用。一个应用程序可使用多个DLL文件，一个DLL文件也可能被不同的应用程序使用，这样的DLL文件被称为共享DLL文件。
  DLL文件中存放的是各类程序的函数(子过程)实现过程，当程序需要调用函数时需要先载入DLL，然后取得函数的地址，最后进行调用。使用DLL文件的好处是程序不需要在运行之初加载所有代码，只有在程序需要某个函数的时候才从DLL中取出。另外，使用DLL文件还可以减小程序的体积。

##（二）训练结果评价指标：
![指标](https://s3.bmp.ovh/imgs/2022/10/14/7dad298b6676f8f0.png)
##（三）报错小结
### 1.直接按字面意思修改
  * *AttributeError: module 'cv2' has no attribute 'COLOR_BGR2gray_frame'*

  * *Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed prs\ops\optimizers\optimizer_v2\adam.py:114: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.*

  * *c:\Users\19131\Desktop\vscode output\机器学习\emotion.py:53: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.*

  * *Traceback (most recent call last):  File "c:\Users\19131\Desktop\vscode output\机器学习\emotion.py", line 74, in <module>*

  * *gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2gray_frame)*

  * *AttributeError: module 'cv2' has no attribute 'COLOR_BGR2gray_frame'**

### 2.GPU加速 dll文件问题：
  * *cudart64_110.dll not found sys
cublas64_11.dll not found sys
cublasLt64_11.dll not found sys
cufft64_10.dll not found bin
curand64_10.dll not found
cusolver64_11.dll not found sys
cusparse64_11.dll not found bin
cudnn64_8.dll not found*
  **Solution：把"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin里的dll放在了"C:\Windows\System"下面，但其实这两个路径都已经在Path的环境变量里面了，
  为什么？？**
[CUDA文件缺失解决方法——以cudart64_110.dll not found为例][21]

## （四）识别成果
> **见b站视频：**[表情识别，但是只因][28]

![只因](https://s3.bmp.ovh/imgs/2022/10/14/d05ec61291c5f786.png)
## （五）train.py的卷积神经网络&两个库

### 1. keras
keras中的主要数据结构是model（模型），它提供定义完整计算图的方法。通过将图层添加到现有模型/计算图，我们可以构建出复杂的神经网络。

#### 1.1 Sequential模型
> Keras有两种不同的构建模型的方法：
Sequential models、Functional API

`Sequential`字面上的翻译是顺序模型，更准确的应该理解为堆叠，通过堆叠许多层，构建出深度神经网络，包括全连接神经网络、卷积神经网络(CNN)、循环神经网络(RNN)等等。
如下代码向模型添加一个带有64个大小为3 * 3的过滤器的卷积层:

> from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu'))

#### 1.2 数据处理
`flow_from_directory(directory)`: 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
`directory`: 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用.详情请查看此脚本
`target_size`: 整数tuple,默认为(256, 256). 图像将被resize成该尺寸
`color_mode`: 颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.
`class_mode`: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式。

#### 1.3添加layers（图层）
`Sequential`模型的核心操作是添加layers（图层）.
`model.add(Conv2D(64, (3, 3), activation='relu')) `卷积层
`model.add(MaxPooling2D(pool_size=(2, 2)))` 最大池化层
`model.add(Dense(256, activation='relu'))` 全连接层
`model.add(Dropout(0.5))` dropout层
`model.add(Flatten())` Flattening layer(展平层)

* `Conv2D`
[Conv2D类参数详解][24]
  * `filters`：卷积核（就是过滤器！）的数目（即输出的维度）；卷积核实质上就是权重。
  * `kernel_size`：单个整数或由两个整数构成的list/tuple，卷积核（过滤器）的宽度和长度。（kernel n.核心，要点，[计]内核）。如为单个整数，则表示在各个空间维度的相同长度。
  * `activation`：激励函数

* `MaxPooling2D`:[Keras: GlobalMaxPooling vs. MaxPooling][25]

* `Dropout`:每个输出节点以概率P置0，所以大约每次使用了（1-P）比例的输出。经过交叉验证，隐含节点[dropout率等于0.5的时候效果最好][27]，原因是0.5的时候dropout随机生成的网络结构最多。
  > [过拟合][26] ：简单来说，就是训练的函数不适用于测试集

* `Flatten`层：用来对数组进行展平操作。假设有一张灰度图片，分别是从1到9，则会形成一个新的数组1，2，3，4，5，6，7，8，9。如果是彩色图片的话，它会有3个颜色通道，进行fltten时的步骤也是一样的。

* `fit`:将数据提供给模型。这里还可以指定批次大小（batch size）、迭代次数、验证数据集等等。

### 2. OpenCV
1. `cap = cv2.VideoCapture(0)`参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开
cap = cv2.VideoCapture('./xx.avi')
2. **`ret,frame = cap.read()`**
cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。
其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
3. `bounding_box`：边界框
4. `cv2.CascadeClassifier` 是Opencv中做人脸检测的时候的一个级联分类器。
    > Haar特征是一种反映图像的灰度变化的，像素分模块求差值的一种特征。它分为三类：边缘特征、线性特征、中心特征和对角线特征。用黑白两种矩形框组合成特征模板，在特征模板内用 黑色矩形像素和 减去 白色矩形像素和来表示这个模版的特征值。例如：脸部的一些特征能由矩形模块差值特征简单的描述，如：眼睛要比脸颊颜色要深，鼻梁两侧比鼻梁颜色要深，嘴巴比周围颜色要深等。但矩形特征只对一些简单的图形结构，如边缘、线段较敏感，所以只能描述在特定方向（水平、垂直、对角）上有明显像素模块梯度变化的图像结构。这样就可以进行区分人脸。
[LBP特征原理] [22]

5. `...haarcascade_frontalface_default.xml`Opencv自带训练好的人脸检测模型，这个是人脸检测器。
6. `cv2.cvtColor(p1,p2)` 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
7. `cv2.COLOR_BGR2RGB` 将BGR格式转换成RGB格式
8. `cv2.COLOR_BGR2GRAY` 将BGR格式转换成灰度图片,转换后并不是通常意义上的黑白图片。因为灰度图片并不是指常规意义上的黑白图片。
9. `detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)`对图像进行多尺度检测，一共有8个参数，这里的三个分别意思是输入的图像、[图像的多尺度表示] [23]，第三个不太明白......
10. `cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)`:图像；pt1 和 pt2 参数分别代表矩形的左上角和右下角两个点；shift 参数表示点坐标中的小数位数，shift 为 1 就相当于坐标全部除以 2^1。
11. `np.expand_dims`可以简单理解为扩展数组的形状,作用:
假设你有一张灰度图，读取之后的shape是（360，480）,而模型的输入要求是（1，360，380）或者是（360，480，1）,那么你就可以通过np.expand_dims(a, axis=0)或者np.expand_dims(a, axis=-1)，将形状改变为满足模型的输入。
12. `emotion_model.predict()`:参数为即将要预测的测试集。
13. `cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)`:把文字添加到图像上；要添加的文字；位置；字体；大小；颜色；粗细
14. `cv2.imshow` 
15. `resize` 起到对图片进行缩放的作用。
16. `INTER_CUBIC`
17. `import cv2` 用来放大;
18. `if cv2.waitKey(1) & 0xFF == ord('q')`:
首先，cv2.waitKey(1) & 0xFF将被执行，等待用户按1ms。如果用户按q，那么q的waitKeyreturnDECIMAL VALUE是113。在二进制中，它表示为0b01110001。接下来，执行AND运算符，两个输入分别是0b01110001和0xFF。0b01110001AND0b11111111=0b01110001。确切的结果是DECIMAL VALUE的q。其次，将左表达式0b01110001的值与ord(‘q’)进行比较。显然，这些值与另一个值相同。最后的结果是break被调用。









[0]:https://www.liaoxuefeng.com/wiki/896043488029600/896067008724000
[1]:https://www.bilibili.com/video/BV1MU4y1Y7h5
[2]:https://zhuanlan.zhihu.com/p/412341075
[3]:https://d.jotang.club/t/topic/804/4
[4]:https://blog.csdn.net/u013072756/article/details/124825062
[5]:https://blog.csdn.net/guo_ridgepole/article/details/121031044
[6]:https://blog.csdn.net/weixin_44801799/article/details/123140330
[7]:https://blog.csdn.net/qq_41654985/article/details/108399024?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166555684416782414918508%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166555684416782414918508&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-108399024-null-null.142^v53^pc_rank_34_queryrelevant25,201^v3^control_2&utm_term=sklearn%E6%98%AF%E4%BB%80%E4%B9%88&spm=1018.2226.3001.4187
[8]:https://blog.azurezeng.com/recaptcha-use-in-china/

[18]:https://blog.csdn.net/u012950413/article/details/80376136?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548733916800184187762%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166548733916800184187762&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-80376136-null-null.142^v52^control,201^v3^control_2&utm_term=%E8%BF%87%E6%8B%9F%E5%90%88%E6%98%AF%E4%BB%80%E4%B9%88&spm=1018.2226.3001.4187

[11]:https://blog.csdn.net/weixin_40432828/article/details/82192709

[12]:https://blog.csdn.net/weixin_39784263/article/details/109957071

[13]:https://www.zhihu.com/question/22298352/answer/637156871

[14]:https://www.bilibili.com/video/BV164411b7dx

[15]:https://www.bilibili.com/video/BV1we4y1X7vy

[16]:https://zhuanlan.zhihu.com/p/377513272

[17]:https://www.bilibili.com/video/BV1if4y147hS


[21]: https://blog.csdn.net/weixin_44908427/article/details/124705386?ops_request_misc=&request_id=&biz_id=102&utm_term=cudart64_110.dll%20not%20found&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-124705386.142^v51^control,201^v3^control_2&spm=1018.2226.3001.4187
[22]: http://www.360doc.com/content/18/0202/11/18306241_727150014.shtml
[23]: https://so.csdn.net/so/search?spm=1001.2101.3001.4498&q=.finalThreshold&t=&u=
[24]: https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/121042872?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548399116782391864540%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166548399116782391864540&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-121042872-null-null.142^v52^control,201^v3^control_2&utm_term=Keras%20conv2d%E5%87%BD%E6%95%B0&spm=1018.2226.3001.4187
[25]:https://blog.csdn.net/devil_son1234/article/details/107408356?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548234916800184112629%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166548234916800184112629&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-10-107408356-null-null.142^v52^control,201^v3^control_2&utm_term=MaxPooling2D&spm=1018.2226.3001.4187
[26]:https://blog.csdn.net/u012950413/article/details/80376136?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548733916800184187762%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166548733916800184187762&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-80376136-null-null.142^v52^control,201^v3^control_2&utm_term=%E8%BF%87%E6%8B%9F%E5%90%88%E6%98%AF%E4%BB%80%E4%B9%88&spm=1018.2226.3001.4187
[27]:https://blog.csdn.net/stdcoutzyx/article/details/49022443?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548772616782417097104%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166548772616782417097104&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-49022443-null-null.142^v52^control,201^v3^control_2&utm_term=dropout%E5%8F%82%E6%95%B0&spm=1018.2226.3001.4187
[28]:https://www.bilibili.com/video/BV1eD4y1C7zc