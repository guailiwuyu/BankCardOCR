import cv2
import numpy as np


def rotate_bound(img, angle,des):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    ans = cv2.warpAffine(img, M, (nW, nH))
    cv2.imwrite(des, ans)
'''
image=cv2.imread('../images/test/0000_0000_0000_0000#1.jpg')
angle=5
imag=rotate_bound(image,angle)
cv2.imshow('ww',imag)
cv2.waitKey()
'''

