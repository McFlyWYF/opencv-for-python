'''
函数
    cv2.getTickCount，返回时钟数
    cv2.getTickFrequency，返回时钟频率
'''

'''
使用opencv检测程序效率
'''


import cv2
import numpy as np
import time

e1 = cv2.getTickCount()
e2 = cv2.getTickCount()

time1 = (e2 - e1) / cv2.getTickFrequency()
print(time1)


'''
用窗口大小不同的(5,7,9)的核函数做中值滤波
'''
img1 = cv2.imread('logo.png')
e1 = cv2.getTickCount()
for i in range(5,49,2):
    img1 = cv2.getTickCount()
e2 = cv2.getTickCount()

t = (e2 - e1) / cv2.getTickFrequency()
print(t)


#python中的计算程序运行时间
start_time = time.time()
temp = 0
for i in range(1000):
    temp = temp * i

end_time = time.time()
print(end_time - start_time)

'''
opencv中的默认优化
    cv2.useOptimized()查看优化是否开启
    cv2.setUseOptimized()开启优化
'''


print(cv2.useOptimized())#检查优化是否开启
img = cv2.imread('logo.png')
e1 = cv2.getTickCount()
res = cv2.medianBlur(img,49)
e2 = cv2.getTickCount()
print((e2 - e1) / cv2.getTickFrequency())

cv2.setUseOptimized(False)#关闭优化
e1 = cv2.getTickCount()
res = cv2.medianBlur(img,49)
e2 = cv2.getTickCount()
print((e2 - e1) / cv2.getTickFrequency())