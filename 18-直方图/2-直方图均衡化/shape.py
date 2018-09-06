'''
直方图均衡化就是将集中的像素值横向拉伸。
适用于脸部识别，在进行识别之前，需要将所有的图像均衡化，达到相同的亮度条件。
'''

#使用numpy进行直方图均衡化
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('house.png',0)

#flatten()将数组变成一维
hist,bins = np.histogram(img.flatten(),256,[0,256])

#计算累计分布图
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized,color = 'b')
plt.hist(img.flatten(),256,[0,256],color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc = 'upper left')
plt.show()


#构建numpy掩模数组，cdf为原数组，当数组元素为0时，掩盖，计算时忽略
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

#对被掩盖的元素赋值，赋值为0
cdf = np.ma.filled(cdf_m,0).astype('uint8')
print(cdf)

plt.subplot(122)
plt.plot(cdf_normalized,color = 'b')
plt.hist(img.flatten(),256,[0,256],color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc = 'upper left')
plt.title('均值化之前')

img2 = cdf[img]
plt.subplot(121)
plt.plot(cdf_normalized,color = 'b')
plt.hist(img2.flatten(),256,[0,256],color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc = 'upper left')
plt.title('均值化之后')

plt.show()

'''
opencv中的直方图均衡化
函数：cv2.equalizeHist()，输入仅仅四号一副灰度图像，输出结果是直方图均衡化之后的图像。
'''

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))

cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.subplot(121)
plt.hist(img.ravel(),256,[0,256])
plt.subplot(122)
plt.hist(res.ravel(),256,[0,256])
plt.show()

'''
CLAHE有限对比适应性直方图均衡化
图像被分成很多小块，然后对每一小块分别进行直方图均衡化
'''
img3 = cv2.imread('status.png',0)
equ = cv2.equalizeHist(img3)
res = np.hstack((img3,equ))
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(img3)
cv2.imshow('image',cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()