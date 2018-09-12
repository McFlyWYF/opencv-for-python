import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('shape.png',0)
fast = cv2.FastFeatureDetector_create()

#寻找关键点
#使用了非最大值抑制
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,None,color=(255,0,0))

print('Threshold:',fast.getThreshold())
print('nonmaxSuppression:',fast.getNonmaxSuppression())
print('negihborhood:',fast.getType())
print('Total Keypoints with nonmaxSuppression',len(kp))

#未使用最大值抑制
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print('Total Keypoints without nonmaxSuppression:',len(kp))

img3 = cv2.drawKeypoints(img,kp,None,color=(255,0,0))

plt.subplot(121)
plt.imshow(img2)
plt.subplot(122)
plt.imshow(img3)
plt.show()