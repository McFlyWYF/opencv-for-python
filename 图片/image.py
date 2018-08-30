'''
使用函数cv2.imread()读入图像
cv2.IMREAD_COLOR:读入一副彩色图像
cv2.IMREAD_GRAYSCALE：以灰度图模式读入图像
'''

import numpy as np
import cv2

#读取
img = cv2.imread('hourse.jpg',cv2.IMREAD_COLOR)

'''
显示，第一个参数是窗口的名字
删除特定的窗口，使用cv2.destroyWindow()，括号指定窗口名
'''
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
先创建窗口，再加载图像
'''
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
保存图像
'''
cv2.imwrite('hourse1.png',img)



'''
例子，按下s键保存，按下ESC键退出不保存
'''
img1 = cv2.imread('hourse.jpg',0)
cv2.imshow('image',img1)
k = cv2.waitKey(0)&0xFF#电脑是64位，加上&0xFF
if k == 27:#ESC
    cv2.destroyAllWindows()

elif k == ord('s'):
    cv2.imwrite('hourse_gray.png',img1)
    cv2.destroyAllWindows()


'''
使用matplotlib显示图像
'''
from matplotlib import pyplot as plt

img2 = cv2.imread('hourse.jpg',0)
plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.xticks([])
plt.yticks([])