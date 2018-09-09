'''
Probabilistic Hough Transform是对霍夫变换的优化，不会对每个点进行计算，而是从一幅图像中随机选取一个点集进行计算
函数：cv2.HoughLinesP()
    参数1：minLineLength，线的最短长度
    参数2：MaxLineGap，两条线段之间的最大间隔，如果小于此值，2条直线会看成一条直线。
返回值：直线的起点和终点
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('dave.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize=3)

minLineLength = 100
maxLineGap = 10

lines = cv2.HoughLinesP(edges,1,np.pi / 180,100,minLineLength,maxLineGap)
print(lines)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

plt.imshow(img)
plt.show()