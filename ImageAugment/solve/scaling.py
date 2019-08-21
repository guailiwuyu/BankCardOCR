import cv2
import os
import random
def scaling(img,x,y):
    #  0.6 - 0.9
    height, width = img.shape[:2]
    # 缩小图像
    #0.3 0.5
    #size = (int(width * x), int(height * x))
    #shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # 放大图像
    fx = 1 + x - 0.5
    fy = 1 + y - 0.5
    enlarge = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    return enlarge

#img = cv2.imread('../images/test/0012_6001_0116_229#1.jpg')

def scalingAll(img,dest):
    for i in range(0 ,5):
        size = random.uniform(0.6,0.9)
        #print(size)
        #ans_dir1 = dest + "Scale" + str(i) + ".png"
        ans_dir2 = dest + "Scale" + str(i) + ".png"
        y = scaling(img,size,size)
        #cv2.imwrite(ans_dir1, x)
        cv2.imwrite(ans_dir2, y)




