'''
opencv中的稠密光流
计算图像中的所有点的光流,得到光流的大小和方向，方向对应于H通道，大小对应于V通道。
'''

import cv2
import numpy as np

cap = cv2.VideoCapture('a.avi')

ret,frame = cap.read()
prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame)
hsv[...,1] = 255

while(1):
    ret,frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next,None,0.5,3,15,3,5,1.2,0)

    mag,ang = cv2.cartToPolar(flow[...,0],flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    cv2.imshow('frame3',frame)
    k = cv2.waitKey(30)&0xff
    if k == 27:
        break

    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)

    prvs = next

cap.release()
cv2.destroyAllWindows()