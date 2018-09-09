'''
使用分水岭算法基于掩模的图像分割
函数：cv2.watershed()

原理：对于灰度图像，灰度值高的地方可以看做是山峰，灰度值低的地方可以看做是山谷。向每一个山谷灌不同颜色的水，随着水位的升高，不同山谷的谁就会汇合，为了防止汇合，构建堤坝，不停灌水，直到所有的山峰被淹没，构建的堤坝就是对图像的分割。

最后得到的边界对象的值为-1.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('money.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

plt.imshow(thresh)
plt.xticks([])
plt.yticks([])
plt.show()

#去除图像中的白噪声,使用开运算，去除对象上的空洞使用闭运算。
#去除
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

#确定背景面积
sure_bg = cv2.dilate(opening,kernel,iterations=3)

'''
距离变换的含义是计算一个图像中非零像素点到最近的零像素点的距离。
'''
dist_transform = cv2.distanceTransform(opening,1,5)
ret,sure_fg = cv2.threshold(dist_transform,0.7 * dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


plt.imshow(sure_fg)
plt.xticks([])
plt.yticks([])
plt.show()

#背景标记为0，其他对象从1开始标记。
ret,markers1 = cv2.connectedComponents(sure_fg)
#将不是背景的标记为从1开始的整数
markers = markers1 + 1
#将不确定的区域标记为0
markers[unknown == 255] = 0

markers3 = cv2.watershed(img,markers)
img[markers3 == -1] = [255,0,0]


plt.subplot(121)
plt.imshow(img)
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(markers3)
plt.xticks([])
plt.yticks([])
plt.show()