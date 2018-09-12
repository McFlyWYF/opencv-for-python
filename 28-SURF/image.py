'''
加速稳健特征
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('hourse.jpg',0)
surf = cv2.xfeatures2d.SURF_create(400)

kp,des = surf.detectAndCompute(img,None)

print(len(kp))

#缩减关键点
surf.hessianThreshold = 50000
kp,des = surf.detectAndCompute(img,None)
print(len(kp))

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2)
plt.show()


surf.upright = True
kp = surf.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2)
plt.show()

surf.extended = True
kp,des = surf.detectAndCompute(img,None)
print(surf.descriptorSize())