'''
背景减除：提取我们需要的目标物
使用高斯分布混合对背景像素进行建模，使用这些颜色存在时间的长短作为混合的权重
'''

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret,frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30)&0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()