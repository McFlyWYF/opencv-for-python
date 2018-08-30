import numpy as np
import cv2


img = np.zeros((512,512,3),np.uint8) * 255#背景为白色

#绘制3个圆环

cv2.circle(img,(256,100),60,(0,0,255),-1)
cv2.circle(img,(256,100),25,(0,0,0),-1)

cv2.circle(img,(181,228),60,(0,255,0),-1)
cv2.circle(img,(181,228),25,(0,0,0),-1)

cv2.circle(img,(331,228),60,(255,0,0),-1)
cv2.circle(img,(331,228),25,(0,0,0),-1)

#绘制三角形
t1 = np.array([256,100,219,164,293,164],np.int32)
t1 = t1.reshape((-1,1,2))

t2 = np.array([[181,228],[256,228],[218,164]],np.int32)
t3 = np.array([[331,228],[368,164],[293,164]],np.int32)

#三角形填充
cv2.fillPoly(img,[t1,t2,t3],(0,0,0))

#添加文字
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCv',(130,400),font,2,(255,255,255),5,cv2.LINE_AA)

cv2.imshow('logo',img)
cv2.waitKey(0)
cv2.destroyAllWindows()