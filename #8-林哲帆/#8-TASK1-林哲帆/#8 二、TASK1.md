# 二、TASK1
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


[7]:https://blog.csdn.net/qq_41654985/article/details/108399024?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166555684416782414918508%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166555684416782414918508&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-108399024-null-null.142^v53^pc_rank_34_queryrelevant25,201^v3^control_2&utm_term=sklearn%E6%98%AF%E4%BB%80%E4%B9%88&spm=1018.2226.3001.4187
[8]:https://blog.azurezeng.com/recaptcha-use-in-china/