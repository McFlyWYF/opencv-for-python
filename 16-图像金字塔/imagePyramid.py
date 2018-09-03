'''
函数
    cv2.pyrUp()
    cv2.pyrDown() 从高分辨率图像向上构建金字塔

原理
    一般处理的图像是固定分辨率，但是有些情况需要对同一图像的不同分辨率的子图像处理，
    比如脸，不知道尺寸大小，需要创建一组图像，它们具有不同的分辨率，最大的放下面，最小的放上面，就是图像金字塔。

有2类金字塔：高斯金字塔和拉普拉斯金字塔
    高斯金字塔：顶部是通过将底部图像中的连续的行和列去除得到的。顶部图像的每个像素值等于下一层图像中5个像素的高斯加权平均值，
    操作一次一个MxN的图像变成一个M/2 X N/2的图像，是原来的四分之一。
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

'''
构建一个4层的图像金字塔

下采样
'''
img = cv2.imread('messi5.jpg')
lower_reso1 = cv2.pyrDown(img)
lower_reso2 = cv2.pyrDown(lower_reso1)
lower_reso3 = cv2.pyrDown(lower_reso2)

plt.subplot(221), plt.imshow(img)
plt.title('Orginal image')

plt.subplot(222), plt.imshow(lower_reso1)
plt.title('lower_reso1')

plt.subplot(223), plt.imshow(lower_reso2)
plt.title('lower_reso2')

plt.subplot(224), plt.imshow(lower_reso3)
plt.title('lower_reso3')

plt.show()

'''
构建一个4层的图像金字塔

上采样
    图像的尺寸变大了，但是图像的像素没有变化
'''
higher_reso1 = cv2.pyrUp(lower_reso1)

plt.subplot(121), plt.imshow(lower_reso1)
plt.title('lower_reso1')

plt.subplot(122), plt.imshow(higher_reso1)
plt.title('higher_reso1')

plt.show()

'''
拉普拉斯金字塔可以由高斯金字塔计算得来
公式
    L = Gi - PyrUp(Gi + 1)
    
能够对图像进行最大程度的还原，配合高斯金字塔一起使用.拉普拉斯金字塔则用来从金字塔底层图像中向上採样重建一个图像,用在图像压缩中。
'''

# load an original image
img = cv2.imread('hourse.jpg')

# convert color space from bgr to gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# pyramid level
level = 5

# original image at the bottom of gaussian pyramid
higherResoGauss = img
plt.subplot(2, 1 + level, 1), plt.imshow(higherResoGauss)
plt.title('Gaussian Level ' + '%d' % level)

for l in range(level):

    rows, cols, channels = higherResoGauss.shape

    # delete last odd row of gaussian image
    if rows % 2 == 1:
        higherResoGauss = higherResoGauss[:rows - 1, :]
    # delete last odd column of gaussian image
    if cols % 2 == 1:
        higherResoGauss = higherResoGauss[:, :cols - 1]

    # gaussian image
    lowerResoGauss = cv2.pyrDown(higherResoGauss)
    # even rows and cols in up-sampled image
    temp = cv2.pyrUp(lowerResoGauss)
    print(higherResoGauss.shape, temp.shape)

    # laplacian image
    lowerResoLap = higherResoGauss - temp

    # display gaussian and laplacian pyramid
    plt.subplot(2, 1 + level, l + 2), plt.imshow(lowerResoGauss)
    plt.title('Gaussian Level ' + '%d' % (level - l - 1))
    plt.subplot(2, 1 + level, 1 + level + l + 2), plt.imshow(lowerResoLap)
    plt.title('Laplacian Level ' + '%d' % (level - l - 1))

    higherResoGauss = lowerResoGauss

# display original image and gray image
plt.show()

'''
使用金字塔进行图像融合
步骤：
    1.读入2幅图像，苹果和橘子
    2.构建苹果和橘子的高斯金字塔（6层）
    3.根据高斯金字塔计算拉普拉斯金字塔
    4.在拉普拉斯的每一层进行图像融合
    5.根据融合后的图像金字塔重建原始图像
'''


def sameSize(img1, img2):
    """
    使得img1的大小与img2相同
    """
    rows, cols, dpt = img2.shape
    dst = img1[:rows, :cols]
    return dst


apple = cv2.imread('apple.png')
orange = cv2.imread('orange.png')

# 对apple进行6层高斯降采样
G = apple.copy()
gp_apple = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gp_apple.append(G)

# 对orange进行6层高斯降采样
G = orange.copy()
gp_orange = [G]
for j in range(6):
    G = cv2.pyrDown(G)
    gp_orange.append(G)

# 求apple的Laplace金字塔
lp_apple = [gp_apple[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gp_apple[i])
    L = cv2.subtract(gp_apple[i - 1], sameSize(GE, gp_apple[i - 1]))
    lp_apple.append(L)

# 求orange的Laplace金字塔
lp_orange = [gp_orange[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gp_orange[i])
    L = cv2.subtract(gp_orange[i - 1], sameSize(GE, gp_orange[i - 1]))
    lp_orange.append(L)

# 对apple和orange的Laplace金字塔进行1/2拼接
LS = []
for la, lb in zip(lp_apple, lp_orange):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

# 对拼接后的Laplace金字塔重建获取融合后的结果
ls_reconstruct = LS[0]
for i in range(1, 6):
    ls_reconstruct = cv2.pyrUp(ls_reconstruct)
    ls_reconstruct = cv2.add(sameSize(ls_reconstruct, LS[i]), LS[i])

# 各取1/2直接拼接的结果
r, c, depth = apple.shape
real = np.hstack((apple[:, 0:c // 2], orange[:, c // 2:]))

plt.subplot(221)
plt.imshow(cv2.cvtColor(apple, cv2.COLOR_BGR2RGB))
plt.title("apple")

plt.subplot(222)
plt.imshow(cv2.cvtColor(orange, cv2.COLOR_BGR2RGB))
plt.title("orange")

plt.subplot(223)
plt.imshow(cv2.cvtColor(real, cv2.COLOR_BGR2RGB))
plt.title("real")

plt.subplot(224)
plt.imshow(cv2.cvtColor(ls_reconstruct, cv2.COLOR_BGR2RGB))
plt.title("laplace_pyramid")
plt.show()

cv2.imwrite('real.jpg',real)
cv2.imwrite('laplace_pyramid.jpg',ls_reconstruct)