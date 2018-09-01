'''
腐蚀，膨胀，开运算，闭运算

函数
    cv2.erode()
    cv2.dilate()
    cv2.morphologyEx()

原理
    形态学操作时根据图像形状进行的简单操作，一般对二值化图像进行操作
    两个参数：一是原始图像，二是被称为结构化元素或核
'''


'''
腐蚀，把前景物体的边界腐蚀掉，卷积核移动，如果与对应的原图像的所有像素值都是1，保持中心元素原来的值，否则变为0
可以用来断开两个连在一起的物体。前景图片缩小
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image3.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(erosion)
plt.title('erosion')
plt.xticks([])
plt.yticks([])
plt.show()


'''
膨胀，与卷积核对应的原图像的像素值只要一个为1，中心元素的像素值就是1，会增加白色区域的面积
可以用来连接两个分开的物体。
'''

img = cv2.imread('image3.png',0)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 1)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(dilation)
plt.title('dilation')
plt.xticks([])
plt.yticks([])
plt.show()


'''
开运算
    先进行腐蚀，再进行膨胀
去除噪声
'''

img = cv2.imread('image1.png',0)
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(opening)
plt.title('opening')
plt.xticks([])
plt.yticks([])
plt.show()


'''
闭运算
    先进行膨胀，再进行腐蚀
填充前景物体中的小洞或小黑点
'''

img = cv2.imread('image2.png',0)
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(closing)
plt.title('closing')
plt.xticks([])
plt.yticks([])
plt.show()


'''
形态学梯度
    就是膨胀和腐蚀的差别
    看上去是前景物体的轮廓
'''

img = cv2.imread('image3.png',0)
kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(gradient)
plt.title('gradient')
plt.xticks([])
plt.yticks([])
plt.show()


'''
礼帽
    原始图像与开运算后得到的图像的差
'''

img = cv2.imread('image3.png',0)
kernel = np.ones((5,5),np.uint8)
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(tophat)
plt.title('tophat')
plt.xticks([])
plt.yticks([])
plt.show()

'''
黑帽
    原始图像与闭运算后得到的图像的差
'''

img = cv2.imread('image3.png',0)
kernel = np.ones((5,5),np.uint8)
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)

plt.subplot(121)
plt.imshow(img)
plt.title('Orginal')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(blackhat)
plt.title('blackhat')
plt.xticks([])
plt.yticks([])
plt.show()


#保存得到的图片
cv2.imwrite('erosion.png',erosion)
cv2.imwrite('dilation.png',dilation)
cv2.imwrite('opening.png',opening)
cv2.imwrite('closing.png',closing)
cv2.imwrite('gradient.png',gradient)
cv2.imwrite('tophat.png',tophat)
cv2.imwrite('blackhat.png',blackhat)


'''
结构化元素
    得到一个椭圆形或圆形的核
    函数
        cv2.getStructuringElement()
'''

#矩阵核
a = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#椭圆核
b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

#十字形核
c = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
print(a)
print(b)
print(c)
