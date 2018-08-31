'''
加法，减法，位运算
函数：
    cv2.add()
    cv2.addWeighted()
'''


'''
图像加法
两幅图的大小，类型必须一致
'''
import numpy as np
import cv2

x = np.uint8([250])
y = np.uint8([10])
'''
cv2.add(x,y)如果相加之后的结果大于255，则就是255
x+y 如果相加之后的结果大于255，则对256取模
'''
print(cv2.add(x,y))#250 + 10 = 260 => 255
print(x+y)#250 + 10 = 260 % 256 = 4

'''
图像混合
其实也是加法，不同的是两幅图的权重不同
图像混合计算公式：g(x) = (1 - a)f0(x) + a*f1(x)

进行图像混合时，图片的大小和类型要相同
'''

#dst = a * img1 + β * img2 + γ

img1 = cv2.imread('hourse.jpg')
img2 = cv2.imread('ball.jpg')

dst = cv2.addWeighted(img1,0.4,img2,0.6,0)
from matplotlib import pyplot as plt

plt.imshow(dst)
plt.show()
cv2.imwrite('mix.png',dst)

# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#加法
dst1 = cv2.add(img1,img2)

plt.imshow(dst1)
plt.show()
cv2.imwrite('add.png',dst1)

# cv2.imshow('dst',dst1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
按位运算
    AND,OR,NOT,XOR
'''
'''
将opencv logo放到另一种图片右上角
'''

#加载图像
img3 = cv2.imread('hourse.jpg')
img4 = cv2.imread('logo.png')

#创建roi
rows,cols,channels = img4.shape
roi = img3[0:rows,0:cols]


img4gray = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img4gray,175,255,cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

#put in roi，将要移动的图的像素值设置为白色，其余值为0 黑色，选取不为0的像素值
img3_bg = cv2.bitwise_and(roi,roi,mask = mask)
img4_fg = cv2.bitwise_and(img4,img4,mask = mask_inv)

dst = cv2.add(img3_bg,img4_fg)
img3[0:rows,0:cols] = dst

plt.subplot(231),plt.imshow(img3),plt.title('img3')
plt.subplot(232),plt.imshow(img3_bg),plt.title('img3_bg')
plt.subplot(233),plt.imshow(img4_fg),plt.title('img4_fg')
plt.show()

cv2.imwrite('img3_bg.png',img3_bg)
cv2.imwrite('img4_fg.png',img4_fg)
cv2.imwrite('img3_new.png',img3)


# cv2.imshow('res',img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()