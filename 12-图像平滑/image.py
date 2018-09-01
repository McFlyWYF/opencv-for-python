'''
图像平滑
    使用不同的低通滤波器对图像进行模糊
    使用自定义的滤波器对图像进行卷积(2D卷积)
'''


'''
2D卷积
    低通滤波LPF：去除噪音，模糊图像
    高通滤波HPF：找到图像边缘
    
    函数
        cv.filter2D()
'''

#更新每一个像素值

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('logo.png')
kernel = np.ones((5,5),np.float32) / 25
dst = cv2.filter2D(img,-1,kernel)

cv2.imwrite('image1.png',dst)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(dst)
plt.title('Averaging')
plt.xticks([])
plt.yticks([])
plt.show()



'''
图像模糊(12-图像平滑)
'''

'''
平均
    用卷积框覆盖区域所有像素的平均值来代替中心元素,这里的卷积框是归一化的
    函数
        cv2.blur()
        cv2.boxFilter()
'''

img = cv2.imread('logo.png')
blur = cv2.blur(img,(5,5))

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(blur)
plt.title('Averaging')
plt.xticks([])
plt.yticks([])
plt.show()


'''
高斯模糊，与卷积核一样，只是里面的值符合高斯分布，方框中心的值最大，其余值递减
原来的求平均变为求加权平均数，高斯核大小必须是奇数
函数
    cv2.GaussianBlur()
'''

img = cv2.imread('logo.png')

#0是指根据窗口大小(5,5)来计算高斯函数标准差
blur = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(blur)
plt.title('Gaussian')
plt.xticks([])
plt.yticks([])
plt.show()


'''
中值模糊，用中心像素的周围的值取代中心像素的值,这个滤波器经常用来去除椒盐噪声
'''

#先给原图加上噪声，再使用中值模糊

#给原图加上椒盐噪声
def saltpepper(img,n):
    m=int((img.shape[0]*img.shape[1])*n)
    for a in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=255
        elif img.ndim==3:
            img[j,i,0]=255
            img[j,i,1]=255
            img[j,i,2]=255
    for b in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=0
        elif img.ndim==3:
            img[j,i,0]=0
            img[j,i,1]=0
            img[j,i,2]=0
    return img
img = cv2.imread('logo.png')
saltImage = saltpepper(img,0.5)

median = cv2.medianBlur(saltImage,5)
cv2.imwrite('median1.png',saltImage)
cv2.imwrite('median2.png',median)

plt.subplot(121)
plt.imshow(saltImage)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(median)
plt.title('median')
plt.xticks([])
plt.yticks([])
plt.show()


'''
双边滤波
    使用函数cv2.bilateralFilter()在保持边界清晰的情况下去除噪声
'''

img = cv2.imread('image.png')

#0是指根据窗口大小(5,5)来计算高斯函数标准差
blur = cv2.bilateralFilter(img,9,75,75)

#纹理被模糊了，但是边界还在

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(blur)
plt.title('Gaussian')
plt.xticks([])
plt.yticks([])
plt.show()