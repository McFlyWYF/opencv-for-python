'''
找轮廓，绘制轮廓
函数
    cv2.findContours()
    cv2.drawContours()
'''

'''
轮廓可以简单认为是将连续的点连在一起的曲线，具有相同的颜色或者灰度。
    1.使用二值化图像，在寻找轮廓之前，进行阈值化处理或者Canny边界检测。
    2.查找轮廓的函数会修改原始图像，将原始图像保存在其他变量中。
    3.要找的物体是白色，背景是黑色。
    
函数cv2.findContours()有3个参数：
    1.输入图像
    2.轮廓检索模式
    3.轮廓近似方法
返回值有3个：
    1.图像
    2.轮廓
    3.轮廓的层析结构
    
每一个轮廓都是一个numpy数组，包含对象边界点(x,y)的坐标
'''

''''
函数cv2.drawContours()可以绘制轮廓，可以根据提供的边界点绘制任何形状。
有3个参数：
    1.原始图像
    2.轮廓，一个列表
    3.轮廓的索引，设置-1时是绘制所有轮廓
    4.颜色
    5.厚度
'''

#在一幅图像上绘制所有轮廓
import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('logo.png')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(contours)
print(hierarchy)

#绘制所有轮廓
img1 = cv2.drawContours(img,contours,-1,(0,0,255),3)

#绘制独立轮廓
img2 = cv2.drawContours(img,contours,5,(0,255,0),3)


plt.subplot(131)
plt.imshow(img1)
plt.subplot(132)
plt.imshow(img2)
plt.subplot(133)
plt.imshow(image)

plt.show()


'''
轮廓的近似方法
    cv2.CHAIN_APPROX_NONE储存所有的边界点
    cv2.CHAIN_APPROX_SIMPLE将轮廓的冗余点去掉，压缩轮廓
'''

img3 = cv2.imread('image.png')
imgray1 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(imgray1,127,255,0)

image1,contours1,hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
image2,contours2,hierarchy2 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

img4 = cv2.drawContours(img3,contours1,-1,(0,0,255),3)
img5 = cv2.drawContours(img3,contours2,-1,(0,0,255),3)

plt.subplot(121)
plt.imshow(img4)
plt.subplot(122)
plt.imshow(img5)
plt.show()