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


[21]: https://blog.csdn.net/weixin_44908427/article/details/124705386?ops_request_misc=&request_id=&biz_id=102&utm_term=cudart64_110.dll%20not%20found&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-124705386.142^v51^control,201^v3^control_2&spm=1018.2226.3001.4187
[22]: http://www.360doc.com/content/18/0202/11/18306241_727150014.shtml
[23]: https://so.csdn.net/so/search?spm=1001.2101.3001.4498&q=.finalThreshold&t=&u=
[24]: https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/121042872?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548399116782391864540%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166548399116782391864540&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-121042872-null-null.142^v52^control,201^v3^control_2&utm_term=Keras%20conv2d%E5%87%BD%E6%95%B0&spm=1018.2226.3001.4187
[25]:https://blog.csdn.net/devil_son1234/article/details/107408356?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548234916800184112629%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166548234916800184112629&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-10-107408356-null-null.142^v52^control,201^v3^control_2&utm_term=MaxPooling2D&spm=1018.2226.3001.4187
[26]:https://blog.csdn.net/u012950413/article/details/80376136?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548733916800184187762%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166548733916800184187762&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-80376136-null-null.142^v52^control,201^v3^control_2&utm_term=%E8%BF%87%E6%8B%9F%E5%90%88%E6%98%AF%E4%BB%80%E4%B9%88&spm=1018.2226.3001.4187
[27]:https://blog.csdn.net/stdcoutzyx/article/details/49022443?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166548772616782417097104%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166548772616782417097104&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-49022443-null-null.142^v52^control,201^v3^control_2&utm_term=dropout%E5%8F%82%E6%95%B0&spm=1018.2226.3001.4187
[28]:https://www.bilibili.com/video/BV1eD4y1C7zc