# encoding:utf-8
import cv2 as cv
import numpy as np
import os
import random
import imageio
from PIL import Image

import binaryzation
import denoise
import eightWays
import sys
import Gaublur
import binary
import eighttest
import PCAJittering
import rotate
import addLine
import randomColor
import scaling


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        # print(path + ' 创建成功')
        return True
    else:
        # print(path + ' 目录已存在')
        return False



# 创建图片文件夹
def mkd(path,des):
    img_list = os.listdir(path)
    img_num = len(img_list)
    for i in range(img_num):
        index = img_list[i].rfind('.')
        name = img_list[i][:index]
        mkdir(des + name)

def solveStep1(path,des_path):
    img_list = os.listdir(path)
    img_num = len(img_list)
    # 存放位置   imgpath + img_list[i]
    for i in range(img_num):
        img_path = os.path.join(path, img_list[i])
        index = img_list[i].rfind('.')
        name = img_list[i][:index]

        print(img_path)
        print(name)
        print(img_list[i])

        src = cv.imread(img_path)
        img = Image.open(img_path)

        tempdir = des_path + '/' + name + '/'
        binary.binaryFunc(img_path, tempdir + "allBinary.png",
                          tempdir +  "localBinary.png",
                          tempdir +  "Gray.png")
        print("二值化")

        randomColor.randomAll(img, tempdir)
        print("随机颜色")

        addLine.addLineAll(img, tempdir )
        print("随机线条")

        scaling.scalingAll(src, tempdir )
        print("随机放缩")

        Gaublur.gaussian_noise(src,tempdir  + "Blur1.png",
                               tempdir  + "Blur2.png")
        print("高斯")

        PCAJittering.PCA_Jittering(src, tempdir  + "PCA.png")
        print("特征值提取")

        #eighttest.threadOPS(tempdir, tempdir)
        for num in range(2):
            fileList = os.listdir(tempdir)  # 获取文件路径
            Num = len(fileList)
            for j in range(Num):
                srcc = os.path.join(tempdir, fileList[j])
                imgg = cv.imread(srcc)
                angle = random.randint(-5, 5)
                Index = fileList[j].rfind('.')
                Name = fileList[j][:Index]
                rotate.rotate_bound(imgg,angle,tempdir + "Rotation" + str(num) + Name+".png")

        rename(tempdir)
        '''
        imgGray = img.convert("L")
        denoise.clearNoise(imgGray, 50, 4, 4,imgpath + '/'+ name+'/'+name +"denoise.png")
        print("降噪")
        '''
def rename(path):
    # 去路径 返回文件名
    file_name = os.path.basename(path)
    filelist = os.listdir(path)  # 获取文件路径
    total_num = len(filelist)  # 获取文件长度（个数）
    print(total_num)
    i = 1  # 表示文件的命名是从1开始的
    for item in filelist:
        if item.endswith('.png'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
            src = os.path.join(os.path.abspath(path), item)
            dst = os.path.join(os.path.abspath(path),
                               "img@" + str(i) + '.png')  # 处理后的格式也为png格式的
            try:
                os.renames(src, dst)
                i = i + 1
            except:
                continue

def augment2(orign_path,augment_path):
    mkd(orign_path,augment_path)
    solveStep1(orign_path,augment_path)