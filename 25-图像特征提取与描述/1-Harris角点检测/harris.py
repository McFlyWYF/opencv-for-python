'''
角点检测
函数：cv2.cornerHarris()
    参数1：数据类型为float32的输入图像
    参数2：角点检测考虑的领域大小
    参数3：求导中使用的窗口大小
    参数4：角点检测方程中的自由参数，取值参数为[0.04,0.06]
     cv2.cornerSubPix()，提供亚像素级别的角点检测
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'black.jpg'
img = cv2.imread(filename)
img1 = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

#输入图像是float32，最后一个参数在0,04到0.06之间
dst = cv2.cornerHarris(gray,2,3,0.04)

dst = cv2.dilate(dst,None)

img[dst > 0.01 * dst.max()] = [255,0,0]

plt.subplot(121)
plt.imshow(img1)
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(img)
plt.xticks([])
plt.yticks([])

plt.show()


ret,dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

ret,labels,stats,centroids = cv2.connectedComponentsWithStats(dst)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER,100,0.001)

corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

res = np.hstack((centroids,corners))

#np.int0可以用来省略小数点后面的数字
res = np.int0(res)
img[res[:,1],res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()