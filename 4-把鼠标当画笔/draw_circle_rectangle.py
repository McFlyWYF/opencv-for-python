'''
拖动鼠标，在面板上画圆或矩形
'''

import cv2
import numpy as np


#鼠标按下时为Ture
drawing = False

#如果mode为True，绘制矩阵，按下m变为绘制曲线
mode = True
ix,iy = -1,-1


#创建回调函数
def draw_cricle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    #按下左键时返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    #按下左键并移动是绘制图形，event可以查看移动，flag查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)

            else:
                #绘制圆圈
                cv2.circle(img,(x,y),3,(0,0,255),-1)

    #当鼠标松开停止绘画
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_cricle)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode

    elif k == 27:
        break