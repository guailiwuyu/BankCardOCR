# -*- coding:utf-8 -*-
import cv2
from math import *
import numpy as np
import time,math
import os
import re
 
'''旋转图像并剪裁'''
def rotate(
        img,  # 图片
        pt1, pt2, pt3, pt4,
        newImagePath
):
    print(pt1,pt2,pt3,pt4)
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    print(withRect,heightRect)
    angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
    print(angle)
 
    if pt4[1] > pt1[1]:
        print("顺时针旋转")
    else:
        print ("逆时针旋转")
        angle = -angle
 
    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]   # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
 
    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
 
    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))
 
    # 处理反转的情况
    if pt2[1] > pt4[1]:
        pt2[1],pt4[1] = pt4[1],pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0],pt3[0] = pt3[0],pt1[0]
 
    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    cv2.imwrite(newImagePath, imgOut)  # 裁减得到的旋转矩形框
    return imgRotation  # rotated image

# def test(directory,imageName):
#     y = "107 532 1134 487 111 617 1137 571"
#     items = y.split(' ')
#     pt1 = [int(items[0]), int(items[1])]
#     pt2 = [int(items[4]), int(items[5])]
#     pt3 = [int(items[6]), int(items[7])]
#     pt4 = [int(items[2]), int(items[3])]
#     imgSrc = cv2.imread(imageName)
#     rotate(imgSrc,pt1,pt2,pt3,pt4,directory+ 'crop_'+imageName)
 
# if __name__=="__main__":
#     directory = "D://"
#     imageName = "7.jpg"
#     test(directory,imageName)