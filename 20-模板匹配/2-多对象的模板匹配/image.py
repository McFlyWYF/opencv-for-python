'''
多对象的模板匹配，需要使用阈值
'''

#找到其中所有的硬币

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_rgb = cv2.imread('Mario.png')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template = cv2.imread('money.png',0)
w,h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
print(res)
threshold = 0.8#设置的阈值

loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb,pt,(pt[0] + w,pt[1] + h),(0,0,255),1)

plt.subplot(121)
plt.imshow(img_gray)
plt.title('Before')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(img_rgb)
plt.title('After')
plt.xticks([])
plt.yticks([])
plt.show()