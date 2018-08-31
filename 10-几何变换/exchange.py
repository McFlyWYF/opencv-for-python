'''
移动，旋转，仿射变换

函数
    cv2.getPerspectiveTransform

    两种变换函数
        cv2.warpAffine  参数是2x3的变换矩阵
        cv2.warpPerspective   参数是3x3的变换矩阵
'''

'''
扩展缩放
    改变图像的尺寸大小
        函数
            cv2.resize()
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('logo.png')

res = cv2.resize(img,None,fx = 2,fy = 2,interpolation=cv2.INTER_CUBIC)

height,width = img.shape[:2]
res = cv2.resize(img,(2 * width,2 * height),interpolation=cv2.INTER_CUBIC)

cv2.imwrite('res1.png',res)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(res)
plt.show()

'''
平移
    平移就是将对象换一个位置，沿某个方向进行移动

移动矩阵
   H = [ 1  0  x ]
       [ 0  1  y ]

移动（100,50）
'''

img1 = cv2.imread('logo.png')
a = np.float32([[1,0,100],[0,1,50]])
rows,cols = img.shape[:2]

res1 = cv2.warpAffine(img,a,(rows,cols))#图像，变换矩阵，变换后大小
cv2.imwrite('res2.png',res1)
plt.subplot(121)
plt.imshow(img1)
plt.subplot(122)
plt.imshow(res1)
plt.show()


'''
旋转

    对一个图像旋转，需要用到旋转矩阵
        H = [ cosθ  -sinθ ]
            [ sinθ   cosθ ]
            
            
    cv2.getRotationMatrix2D()
'''

#旋转90度



img2 = cv2.imread('logo.png')
rows,cols = img2.shape[:2]
#第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
#第三个参数：变换后的图像大小
res = cv2.warpAffine(img2,M,(rows,cols))
cv2.imwrite('res3.png',res)

plt.subplot(121)
plt.imshow(img2)
plt.subplot(122)
plt.imshow(res)
plt.show()


'''
仿射变换
    图像的旋转加上拉升就是图像仿射变换
    
    使用cv2.getAffineTransform(pos1,pos2)得到仿射矩阵
'''


img3 = cv2.imread('logo.png')
rows,cols = img.shape[:2]
pst1 = np.float32([[50,50],[200,50],[50,200]])
pst2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pst1,pst2)

#第三个参数，变换后的图像大小
res = cv2.warpAffine(img3,M,(rows,cols))
cv2.imwrite('res4.png',res)

plt.subplot(121)
plt.imshow(img3)
plt.subplot(122)
plt.imshow(res)
plt.show()


'''
透视变换

    透视需要的是一个3*3的矩阵，cv2.getPerspectiveTransform(pts1,pts2)得到矩阵
    函数cv2.warpPerspective(img,M,(200,200))进行
'''

img4 = cv2.imread('logo.png')
rows,cols = img4.shape[:2]

pts1 = np.float32([[56,65],[238,52],[28,237],[239,240]])
pts2 = np.float32([[0,0],[200,0],[0,200],[200,200]])

M = cv2.getPerspectiveTransform(pts1,pts2)
res = cv2.warpPerspective(img4,M,(300,300))
cv2.imwrite('res5.png',res)

plt.subplot(121)
plt.imshow(img4)
plt.subplot(122)
plt.imshow(res)
plt.show()
