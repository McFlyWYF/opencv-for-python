'''
直线
'''

import cv2
import numpy as np


img = np.zeros((512,512,3),np.uint8)

#坐标(100,100)到(300,300)，2代表线条粗细
cv2.line(img,(100,100),(300,300),(255,0,0),2)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()