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

[4]:https://blog.csdn.net/u013072756/article/details/124825062

[5]:https://blog.csdn.net/guo_ridgepole/article/details/121031044

[6]:https://blog.csdn.net/weixin_44801799/article/details/123140330


