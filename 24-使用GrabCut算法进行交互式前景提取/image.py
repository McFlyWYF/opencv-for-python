'''
GrabCut算法原理，使用GrabCut算法提取图像的前景，创建一个交互式程序完成前景提取
'''

'''
步骤：
    1.输入一个矩形，矩形外的是背景，矩形内的是未知的
    2.对输入图像做一个初始化标记，标记前景和背景像素
    3.使用高斯混合模型对前景和背景建模
    4.根据与已知分类的像素的关系进行分类
    5.直到收敛
'''

'''
函数：cv2.grabCut()
    参数1：输入图像
    参数2：掩模图像，用来确定哪些区域是背景，前景
    参数3：包含前景的矩形，格式为(x,y,w,h)
    参数4：算法内部使用的数组，创建2个大小为(1,65)，数据类型为np.float64的数组
    参数5：算法的迭代次数
    参数6：用来确定进行修改的方式，矩形模式或者掩模模式
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('messi5.jpg')
#掩模
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
#函数的返回值是更新的mask，bgdModel,fgdModel
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0),0,1).astype('uint8')
img = img * mask2[:,:,np.newaxis]

cv2.imwrite('messi.jpg',img)
plt.imshow(img)
plt.colorbar()
plt.show()

newmask = cv2.imread('messi.jpg',0)

mask[newmask == 0] = 0
mask[newmask == 255] = 1

mask,bgdModel,fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

mask = np.where((mask ==2) | (mask == 0),0,1).astype('uint8')
img = img * mask[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()