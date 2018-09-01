'''
简单阈值，自适应阈值，Otsu's二值化
函数
    cv2.threshold()，第一个参数是原图像，原图像应该是灰度图，
                    第二个参数是用来对像素值进行分类的阈值。
                    第三个参数是当像素值高于阈值时应该赋予一个新的像素值
                    第四个参数是阈值方法
    cv2.adaptiveThreshold()三个参数
'''


'''
简单阈值
    当像素值高于阈值时，给像素赋予一个新值，否则赋予另外一种颜色
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('gray.jpg',0)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()



'''
自适应阈值
    在一幅图的不同区域采用不同的阈值
    
    第一个参数：指定阈值计算方法
        cv2.ADPTIVE_THRESH_MEAN_C：阈值取自相邻区域的平均值
        cv2.ADPTIVE_THRESH_GAUSSIAN_C：阈值取自相邻区域的加权和，权重为一个高斯窗口
        
    第二个参数：邻域大小（计算阈值的区域大小）
    第三个参数：
        常数，阈值等于平均值或者加权平均值减去常数
'''


img = cv2.imread('hourse.jpg',0)
# 中值滤波
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#11 为邻域大小，2为C值
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
titles = ['Original Image','Global Thresholding(v=127)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']

images = [img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()


'''
Otsu's二值化
    简单来说就是对一副双峰图像自动根据直方图计算出一个阈值
    
    第一种方法：
        设置127为全局阈值
    第二种方法：
        使用Otsu二值化
    第三种方法：
        使用5x5的高斯核去除噪音，再使用Otsu二值化
'''


img = cv2.imread('noise.png',0)

#global threshloding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Otsu's thresholding after Gaussian filtering
#(5,5)为高斯核的大小，0为标准差
blur = cv2.GaussianBlur(img,(5,5),0)

#阈值设为0
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#plot all the images and their histograms

images1 = [img,0,th1,
          img,0,th2,
          blur,0,th3]

titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
'Original Noisy Image','Histogram',"Otsu's Thresholding",
'Gaussian filtered Image','Histogram',"Otsu's Thresholdin"]

#这里使用pyplot画直方图的方法plt.hist,参数是一维数组
for i in  range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images1[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images1[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images1[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])

plt.show()


'''
Otsu's二值化算法
    在两个峰之间找到一个阈值，使每一个峰内的方差最小
'''

img = cv2.imread('noise.png',0)
blur = cv2.GaussianBlur(img,(5,5),0)

#计算归一化直方图
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()
bins = np.arange(256)
fn_min = np.inf
thresh = -1


for i in range(1,256):
    p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
    q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
    b1, b2 = np.hsplit(bins, [i])  # weights

    m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
    v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2

    fn = v1 * q1 + v2 * q2
    if fn < fn_min:
        fn_min = fn
        thresh6 = i

ret,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(thresh6,ret)