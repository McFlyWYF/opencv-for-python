'''
opencv中的光流

在连续的两帧之间像素的灰度值是不改变的
相邻的像素具有相同的运动

1.在视频的第一帧图像中检测角点
2使用算法迭代跟踪这些角点
函数如果返回1，就是找到这个点，如果返回0，就是没有找到这个点。将这些参数继续传给函数，迭代下去。
'''


import cv2
import numpy as np

cap = cv2.VideoCapture('a.avi')
feature_params = dict(maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7)

lk_params = dict(winSize = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT))

color = np.random.randint(0,255,(100,3))

ret,old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray,mask = None,**feature_params)

mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30)&0xff
    if k == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
