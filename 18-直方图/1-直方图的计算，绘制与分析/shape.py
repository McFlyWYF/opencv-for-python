'''
计算直方图
绘制直方图
函数：cv2.calcHist(),np.histogram()
直方图的x轴是灰度值;y轴是灰度值点的数目
灰度值的范围是[0,256]
函数：cv2.calcHist()统计一幅图像的直方图
参数1；原图像，格式为uint8或float32，当传入函数时，使用[]括起来。
参数2：灰度图，值为0，彩色图像，参数可以使[0],[1],[2],分别表示B,G,R；用[]括起来。
参数3：掩模图像，统计整幅图设置为None,统计某一部分，需要制作一个掩模图像。
参数4：BIN数目，也就是分组数目，用[]括起来。
参数5：像素值范围，通常为[0,256]。
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

#opencv 统计直方图
img = cv2.imread('flat.png',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])#256 x 1
print(hist.shape)


#numpy统计直方图
hist,bins = np.histogram(img.ravel(),256,[0,256])
hist1 = np.bincount(img.ravel(),minlength=256)#bins是257，因为numpy计算bins的方式为：0-0.99,1-1.99，最后要加一个256
print(hist.shape,bins.shape)
print(hist1.shape)


'''
绘制直方图
1.使用matplotlib绘图函数
2.使用opencv绘图函数
'''

plt.hist(img.ravel(),256,[0,256])
plt.show()

img1 = cv2.imread('dog.png',0)
img2 = cv2.imread('house.png',0)

plt.subplot(221)
plt.imshow(img1)
plt.subplot(222)
plt.hist(img1.ravel(),256,[0,256])

plt.subplot(223)
plt.imshow(img2)
plt.subplot(224)
plt.hist(img2.ravel(),256,[0,256])

plt.show()

'''
多通道直方图
'''

img3 = cv2.imread('flat.png')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img3],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


'''
使用掩模，统计局部区域，将要统计的部分设置为白色，其余为黑色
'''

#创建掩模
mask = np.zeros(img.shape[:2],np.uint8)
mask[100:300,100:400] = 255
masked_img = cv2.bitwise_and(img,img,mask=mask)

hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224)
plt.plot(hist_full)#整幅图的直方图
plt.plot(hist_mask)#掩模后的直方图
plt.xlim([0,256])
plt.show()