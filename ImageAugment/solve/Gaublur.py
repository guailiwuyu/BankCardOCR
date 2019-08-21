#高斯模糊    轮廓还在，保留图像的主要特征  高斯模糊比均值模糊去噪效果好
import cv2 as cv
import numpy as np

def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def gaussian_noise(image,des_dir1,des_dir2):        #加高斯噪声

    dst = cv.GaussianBlur(image, (3, 3), 0)   # 高斯模糊
    cv.imwrite(des_dir1, dst)
    dst1 = cv.GaussianBlur(image, (5, 5), 0)  # 高斯模糊
    cv.imwrite(des_dir2, dst1)



#src = cv.imread('../images/test/0_03b_0.png')
#gaussian_noise(src)

