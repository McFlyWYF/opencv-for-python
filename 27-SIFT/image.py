'''
1.尺度不变特征变换(SIFT)算法
2.在图像中查找SIFT关键点和描述符
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('hourse.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)#找到关键点

img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#绘制关键点

kp,des = sift.compute(gray,kp)

plt.imshow(img)
plt.show()