'''
另一种角点检测技术：Shi-Tomasi角点检测
函数：cv2.goodFeatureToTrack()
输入的是灰度图像，确定检测的角点数目，再设置角点的质量水平，0-1之间，代表角点的最低质量，低于这个数的角点会被忽略，最后再设置两个角点之间的最短欧式距离。
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('luo.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)

#返回的结果是[[311.，250.]]
corners = np.int0(corners)
print(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img)
plt.show()