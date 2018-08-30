import numpy as np
import cv2

'''
如果需要绘制多条直线，polylines比line高效
'''

img = np.zeros((521,521,3),np.uint8)

line1 = np.array([[100,20],[300,20]],np.int32).reshape((-1,1,2))
line2 = np.array([[100,100],[300,100]],np.int32).reshape((-1,1,2))
line3 = np.array([[100,300],[300,300]],np.int32).reshape((-1,1,2))

cv2.polylines(img,[line1,line2,line3],True,(0,255,255),3)#传入的是一个列表

cv2.imshow('lines',img)
cv2.waitKey(0)
cv2.destroyAllWindows()