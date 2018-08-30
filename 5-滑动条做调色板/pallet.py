'''
滑动条绑定到opencv窗口
函数：
    cv2.creatTrackbar()
        参数：
            第一个：滑动条名字
            第二个：滑动条被放置窗口的名字
            第三个：滑动条默认位置
            第四个：滑动条的最大值
            第五个：回调函数
    cv2.getTrackbarPos()
'''

import numpy as np
import cv2

def nothing(x):
    pass


#创建一副黑色图像
img = np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

switch = '0:0FF\n1:ON'#转换按钮，只有为ON时，滑动条才可以使用

cv2.createTrackbar(switch,'image',0,1,nothing)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) &0xFF
    if k == 27:
        break

    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0

    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()
