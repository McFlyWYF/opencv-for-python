'''
图片上绘制文字
参数：
    文字
    位置
    字体类型
    大小

绘制白色的opencv
'''

import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCv',(10,256),font,4,(255,255,255),5,cv2.LINE_AA)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
所有的绘图函数的返回值都是None，不能使用
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
'''