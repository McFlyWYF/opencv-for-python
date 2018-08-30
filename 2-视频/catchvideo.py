'''
使用摄像头捕获视频
cap.read()返回的是一个布尔值，帧读取正确，返回True
使用cap.isOpened()检查是否初始化成功，如果没有，使用cap.open()
cap.get()获取视频的参数信息
'''

import numpy as np
import cv2

cap = cv2.VideoCapture(0)#使用内置摄像头或者可以指定其他摄像头
print(cap.isOpened())

while(True):
    ret,frame = cap.read()#一帧一帧读取
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#转换为灰度图
    print(cap.get(3))#查看每一帧的宽和高
    print(cap.get(4))

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(ret)
        break

#释放
cap.release()
cv2.destroyAllWindows()