import numpy as np
import cv2

'''
获取并修改像素值
'''
img = cv2.imread('hourse.jpg')

# print(img)

px = img[100,100]
print(px)

blue = img[100,100,0]
print(blue)

#修改
img[100,100] = [255,255,255]
print(img[100,100])

#另一种方法,批量修改
print(img.item(10,10,2))
img.itemset((10,10,2),100)
print(img.item(10,10,2))

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''
获取图像属性
    行，列，通道，图像数据类型，像素数目
    img.shape返回的是行，列，通道数的元组
    如果是灰度图，仅返回行和列
'''

img1 = cv2.imread('hourse_gray.png')
print(img1.shape)
print(img.size)#返回图像的像素数目
print(img.dtype)#返回图像的数据类型



'''
图像ROI
将某个区域拷贝到其他区域
'''
img2 = cv2.imread('ball.jpg')
ball = img2[380:440,630:690]
img2[273:333,100:160] = ball
print(img2.shape)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
拆分及合并图像通道
'''
img3 = cv2.imread('ball.jpg')
# b,g,r = cv2.split(img3)#拆分
# img3 = cv2.merge(b,g,r)#合并

'''
修改所有的红色通道的值为0
'''
img3[:,:,2] = 0
print(img3)

'''
为图像扩边（填充）
cv2.copyMakeBorder()
包括如下参数蚋
    • src 输入图像
    • top, bottom, left, right 对应边界的像素数目。
    • borderType 添加哪种类型的边界
        – cv2.BORDER_CONSTANT 添加有颜色的常数值边界
            下一个参数value
        – cv2.BORDER_REFLECT 边界元素的镜像。比如: fedcba|abcdefgh|hgfedcb
        – cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT
            和上面一样，但稍作改动。例如: gfedcb|abcdefgh|gfedcba
        – cv2.BORDER_REPLICATE 重复最后一个元素。例如: aaaaaa|
            abcdefgh|hhhhhhh
        – cv2.BORDER_WRAP 不知䨂怎么䖣了, 就像䦈样: cdefgh|abcdefgh|abcdefg
'''

from matplotlib import pyplot as plt

BLUE = [255,0,0]
img4 = cv2.imread('logo1.png')

replicate = cv2.copyMakeBorder(img4,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img4,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img4,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img4,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img4,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img4,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()