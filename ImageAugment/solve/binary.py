import cv2 as cv
import matplotlib.pyplot as plt


def binaryFunc(image,des_dir1,des_dir2,des_dir3):
    img = cv.imread(image, 0)  # 直接读为灰度图像
    th0 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)  # 换行符号 \
    th1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imwrite(des_dir1, th0)
    cv.imwrite(des_dir2, th1)
    cv.imwrite(des_dir3, img)