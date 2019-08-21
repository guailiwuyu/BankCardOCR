# coding:utf-8
import cv2 as cv
import numpy as np

#全局阈值
def threshold_demo(image,des_dir):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print("threshold value %s"%ret)
    cv.imwrite(des_dir, binary)

#局部阈值
def local_threshold(image,des_dir):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary =  cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10)
    #cv.namedWindow("binary1" )
    #cv.imshow("binary1", binary)
    cv.imwrite(des_dir, binary)

#用户自己计算阈值
def custom_threshold(image,des_dir):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    print("mean:",mean)
    ret, binary =  cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imwrite(des_dir, binary)


#cv.namedWindow('input_image') #设置为WINDOW_NORMAL可以任意缩放
#cv.imshow('input_image', src)

'''
src = cv.imread('../images/test/0_03b_0.png')
#调用函数
threshold_demo(src)
local_threshold(src)
custom_threshold(src)
cv.waitKey(0)
cv.destroyAllWindows()
'''
