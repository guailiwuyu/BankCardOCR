# BankCardOCR
A bank cardnumber recognition system based on deep learning algorithm.

一、开发和运行系统

windows10 系统；


二、开发工具和软件介绍

1.pyCharm; 

2.python3.5;

3.tensorflow1.13.1 / tensorflow-gpu1.13.1,cuDNN7.4,CUDA10.0;

4.Jupyter Notebook;

5.WebStorm


三、运行环境配置：

1.安装 python3.5 及以上版本，并配置系统环境变量，使得可以通过 pip 命令安装三方库，使得可以通过 cmd 命令行执行 python 命令；

2.pip 命令安装 python 三方库插件：

tensorflow   1.13.1 或者 tensorflow-gpu （ 1.13.1 ）、 cuDNN （ 7.4 ）、 CUDA （ 10.0 ）

matplotlib    3.0.3

easydict	  1.9

pyyaml	5.1

opencv_python 4.0.0.21

scipy	    1.2.1

pillow	    6.0.0

numpy	    1.16.1

imageio	    2.5.0

flask      	    1.0.3

可以通过豆瓣镜像网站加速安装下载三方库，例如安装 tensorflow-gpu 示例：
pip install tensorflow-gpu -i https://pypi.doubanio.com/simple/ 3.运行步骤
运行项目时，可以在当前目录，即 demo 文件夹下用 cmd 命令行或 powershell 命令行运行 python demo.py。
程序自动读取 test_images 的图片进行卡号定位和识别， 框出的定位结果保存在test_result 文件夹下，识别的卡号结果保存在 test_result/result.txt 文件里，test_result2 文件夹存放卡号截取的图片，用于卡号识别读取。

卡号定位的模型保存在 card_number_detection/checkpoints 文件夹里， 卡号识别的模型保存在 card_number_recognition/save 文件夹里
