'''
多边形
'''

import numpy as np
import cv2

'''
需要指出每个顶点的坐标，用点的坐标构建一个大小等于行数X1X2的数组，
行数就是点的数目，数组数据类型是int32
'''
img = np.zeros((512,512,3),np.uint8)

pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pts = pts.reshape((-1,1,2))#如果第三个参数是False,得到的多边形是不闭合的

cv2.polylines(img,pts,True,(255,0,0),5)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
