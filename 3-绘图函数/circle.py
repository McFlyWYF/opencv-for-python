import cv2
import numpy as np

'''
画圆
'''

img = np.zeros((512,512,3),np.uint8)

#指定中心坐标和半径
cv2.circle(img,(256,256),63,(0,0,255),-1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()