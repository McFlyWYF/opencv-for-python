'''
Canny边缘检测
函数
    cv2.Canny()

原理
    噪声去除：第一步使用5x5的高斯滤波器去除噪声
    计算图像梯度：对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数，根据这2副图找到边界的梯度和方向
    梯度的方向一般与边界垂直，梯度方向被归为4类：垂直，水平，两个对角线
    非极大值抑制：对图像扫描，去除非边界的点，对每一个像素检查，看这个点的梯度是不是周围具有相同梯度方向的点中最大的
    滞后阈值：确定边界，设置2个阈值，minVal和maxVal，当图像的灰度梯度高于maxVal时是边界，低于minVal的被抛弃，介于两者之间的，看这个点是否和真正的边界点相连

'''

'''
使用函数cv2.canny()
    参数1：输入 图像
    参数2；minVal
    参数3：maxVal
    参数4：设置图像梯度Sobel卷积核的大小，默认值是3
    参数5：设定求梯度大小的方程，如果为True，使用根号(G(x)^2 + G(y)^2)，否则使用|G(x)^2| + |G(y)^2|，默认值是False
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('messi5.png',0)
edges1 = cv2.Canny(img,100,200)
edges2 = cv2.Canny(img,100,200,True)

plt.subplot(131),plt.imshow(img,cmap='gray')
plt.title('Orginal Image'),plt.xticks([]),plt.yticks([])

plt.subplot(132),plt.imshow(edges1,cmap='gray')
plt.title('Edge Image one'),plt.xticks([]),plt.yticks([])

plt.subplot(133),plt.imshow(edges2,cmap='gray')
plt.title('Edge Image two'),plt.xticks([]),plt.yticks([])

cv2.imwrite('edges1.png',edges1)

plt.show()


#设置一个滑动条来调节阈值
def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('min','image',0,1000,nothing)
cv2.createTrackbar('max','image',0,1000,nothing)

img = cv2.imread('messi5.png',cv2.IMREAD_GRAYSCALE)

while(1):
    r = cv2.getTrackbarPos('min','image')
    g = cv2.getTrackbarPos('max','image')

    edges3 = cv2.Canny(img,r,g)

    cv2.imshow('image',edges3)
    k = cv2.waitKey(1) &0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cv2.waitKey(0)

