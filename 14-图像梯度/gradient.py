'''
图像梯度
图像边界
函数
    cv2.Sobel()
    cv2.Schar()
    cv2.Laplacian()

原理
    梯度简单来说就是求导，梯度滤波器：Sobel求一阶或二阶导数,Schar求一阶或二阶导数,Laplacian求二阶导数
'''


'''
Sobel算子和Schar算子
    Sobel算子是高斯平滑与微分操作的结合，如果卷积核的大小Ksize = -1,会使用3x3的Schar滤波器
    3x3 Schar滤波器卷积核：
    X方向：-3  0  3
          -10  0 10
          -3  0  3
          
    Y方向: -3  -10  -3
           0   0    0
           3   10   3
    
    也就是对方向的转置
'''

'''
Laplacian算子
    拉普拉斯算子使用二阶导数的形式定义，计算拉普拉斯算子时直接调用Sobel算子
    
    滤波器卷积核
        k = 0  1  0
            1  -4 1
            0  1  0
'''

'''
使用3种方法进行操作，卷积核大小5X5
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('dave.jpg',0)

#cv2.CV_64F输出图像的深度，可以使用-1，与原图像保持一致，np.uint8
laplacian = cv2.Laplacian(img,cv2.CV_64F)

#参数1,0为只在X方向求一阶导数，最大可以求二阶
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

#参数0,1为只在y方向求一阶导数，最大可以求二阶
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Orginal'),plt.xticks([]),plt.yticks([])

plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'),plt.xticks([]),plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('sobelx'),plt.xticks([]),plt.yticks([])

plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('sobely'),plt.xticks([]),plt.yticks([])

cv2.imwrite('laplacian.jpg',laplacian)
cv2.imwrite('sobelx.jpg',sobelx)
cv2.imwrite('sobely.jpg',sobely)

plt.show()

'''
黑到白的边界的导数是整数，白到黑的边界的导数是负数，如果原图像的深度是np.int8时，负值会被截断为0，就是把边界丢失掉
下面的代码是深度不同的不同效果
'''

img1 = cv2.imread('boxs.png',0)

#output dtype =cv2.CV_8U
sobelx8u = cv2.Sobel(img1,cv2.CV_8U,1,0,ksize=5)

#output dtype = cv2.CV_64F and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
#取绝对值
abs_sobel64f = np.absolute(sobelx64f)
#转换为8u
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img1,cmap = 'gray')
plt.title('Orginal'),plt.xticks([]),plt.yticks([])

plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'),plt.xticks([]),plt.yticks([])

plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'),plt.xticks([]),plt.yticks([])


cv2.imwrite('CV_8U.png',sobelx8u)
cv2.imwrite('abs(CV_64F).png',sobel_8u)

plt.show()

'''
图像显示的是从黑到白的边界存在，从白到黑的边界不存在，导致边界消失
所以先求CV_64F，再对其取绝对值，再转化为CV_8U
'''

plt.subplot(1,3,1),plt.imshow(sobelx64f,cmap = 'gray')
plt.title('Sobel CV_64F'),plt.xticks([]),plt.yticks([])

plt.subplot(1,3,2),plt.imshow(abs_sobel64f,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'),plt.xticks([]),plt.yticks([])

plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel CV_8U'),plt.xticks([]),plt.yticks([])

plt.show()


