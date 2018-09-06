'''
2D直方图，图像的2个特征：颜色和饱和度
'''

'''
opencv中的2D直方图
函数：cv2.calcHist()
计算一维直方图，绘制颜色直方图，需要将图像的颜色空间从BGR转换到HSV
计算2D直方图。处理H,S两个通道
H的通道为180，S的通道为256
H的取值范围在0到180，S的取值范围在0到256
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('flat.png')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
print(hist)


'''
numpy中的2D直方图
函数：np.histogram2d()
参数1：H通道
参数2：S通道
参数3：bins的数目
参数4：数值范围
'''
plt.imshow(hist,interpolation='nearest')
plt.xlabel('S')
plt.ylabel('H')
plt.show()

