'''
使用Opencv处理鼠标事件
函数 cv2.setMouseCallback()
'''


#查看所有被支持的鼠标事件
import cv2
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

'''
案例：
    在点击过的地方绘制一个圆圈
'''

import numpy as np

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:#双击
        cv2.circle(img,(x,y),20,(255,0,0),-1)

#创建图像与窗口并将窗口与回调函数绑定
img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:#ESC
        break

cv2.destroyAllWindows()

