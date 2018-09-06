'''
直方图反向投影，可以用来做图像分割，或者找到感兴趣的部分。
输出的值代表输入图像上对应点属于目标对象的概率，
输出图像中的像素值越高（越白）的带你就代表要搜索的目标
先对原图进行分割，分割出我们要找的目标，然后对目标进行颜色直方图，接着把颜色直方图投影到输入图像中寻找目标，最后设置适当的阈值对概率图像进行二值化

函数：cv2.calcBackProject()
参数1 目标直方图
'''

'''
numpy中的算法
创建两幅颜色直方图，目标图像的直方图 M，输入图像的直方图 I
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

roi = cv2.imread('yellow.png')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

#搜索目标图像,黄色衣服
target = cv2.imread('ball.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

roihist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])

#归一化
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

#卷积把分散的点连在一起
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dst = cv2.filter2D(dst,-1,disc)

#threshold and AND
ret,thresh = cv2.threshold(dst,50,255,0)

#3通道
thresh = cv2.merge((thresh,thresh,thresh))

#按位操作
res = cv2.bitwise_and(target,thresh)

res = np.hstack((target,thresh,res))

plt.imshow(res,cmap='gray')
plt.show()