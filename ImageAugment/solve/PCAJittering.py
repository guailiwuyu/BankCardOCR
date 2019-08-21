import numpy as np
import os
from PIL import Image, ImageOps
import random
from scipy import misc
import imageio
import cv2 as cv

def PCA_Jittering(src,des_dir):
    img = Image.fromarray(cv.cvtColor(src, cv.COLOR_BGR2RGB))

    img = np.asanyarray(img, dtype = 'uint8')

    img = img / 255.0
    img_size = img.size // 3    #转换为单通道
    img1 = img.reshape(img_size, 3)

    img1 = np.transpose(img1)   #转置
    img_cov = np.cov([img1[0], img1[1], img1[2]])    #协方差矩阵
    lamda, p = np.linalg.eig(img_cov)     #得到上述协方差矩阵的特征向量和特征值

    #p是协方差矩阵的特征向量
    p = np.transpose(p)    #转置回去

    #生成高斯随机数********可以修改
    alpha1 = random.gauss(0,3)
    alpha2 = random.gauss(0,3)
    alpha3 = random.gauss(0,3)

    #lamda是协方差矩阵的特征值
    v = np.transpose((alpha1*lamda[0], alpha2*lamda[1], alpha3*lamda[2]))     #转置

    #得到主成分
    add_num = np.dot(p,v)

    #在原图像的基础上加上主成分
    img2 = np.array([img[:,:,0]+add_num[0], img[:,:,1]+add_num[1], img[:,:,2]+add_num[2]])

    #现在是BGR，要转成RBG再进行保存
    img2 = np.swapaxes(img2,0,2)
    img2 = np.swapaxes(img2,0,1)
    imageio.imsave(des_dir,img2)



