'''
查找轮廓的不同特征：面积，周长，重心，边界框
'''

'''
矩：图像的矩可以计算图像的质心，面积
函数：cv2.moments()计算的矩以字典形式返回
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test1.png',0)
ret,thresh = cv2.threshold(img,127,255,0)
image,contours,hierarchy = cv2.findContours(thresh, 1,2)
cnt = contours[0]
M = cv2.moments(cnt)
print (M)

#计算重心
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

print('坐标x:',cx)
print('坐标y:',cy)

'''
轮廓面积
函数
    cv2.contourArea()
'''

area = cv2.contourArea(cnt)
print('面积:',area)

'''
轮廓周长
函数
    cv2.arcLength(),第二个参数可以用来指定对象的形状是闭合的(true)还是打开的。
'''
perimeter1 = cv2.arcLength(cnt,True)
perimeter2 = cv2.arcLength(cnt,False)
print('闭合周长：',perimeter1)
print('打开周长：',perimeter2)

'''
轮廓近似
函数
    cv2.approxPolyDP()第二个参数是原始轮廓到近似轮廓的最大距离，是一个准确度参数
'''

img1 = cv2.imread('test.png',0)
ret,thresh1 = cv2.threshold(img1,127,255,0)
image,contours1,hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt1 = contours1[0]

epsilon1 = 0.1 * cv2.arcLength(cnt1,True)
epsilon2 = 0.01 * cv2.arcLength(cnt1,True)

apprxo1 = cv2.approxPolyDP(cnt1,epsilon1,True)#第三个参数是设定弧线是否闭合
apprxo2 = cv2.approxPolyDP(cnt1,epsilon2,True)#第三个参数是设定弧线是否闭合

img2 = cv2.drawContours(img1,apprxo1,-1,(255,0,0),3)
img3 = cv2.drawContours(img1,apprxo2,-1,(255,0,0),3)

plt.subplot(121)
plt.imshow(img2)
plt.title('10% epsilon')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(img3)
plt.title('1% epsilon')
plt.xticks([])
plt.yticks([])

plt.show()


'''
凸包
函数：cv2.convexHull()可以检测一个曲线是否有凸性缺陷，并能纠正
points 我们要传入的轮廓
hull 输出
clockwise 方向标志，设置为True，输出的凸包是顺时针，否则为逆时针
returnPoints默认值为True,会返回凸包上点的坐标，设置为False，返回与凸包点对应的轮廓上的点。
'''

# hull = cv2.convexHull(points[,hullp,clockwise[,returnPoints]])
hull1 = cv2.convexHull(cnt1,returnPoints=True)
hull2 = cv2.convexHull(cnt1,returnPoints=False)

print(hull1)#返回的是坐标
print(hull2)#返回的是索引
print(cnt1[571])
print(cnt1[286])
print(cnt1[0])
print(cnt1[770])

#如果要获得凸性缺陷，需要把returnPoints设置为False，


'''
凸性检测
函数:cv2.isContourConvex()检测一个曲线是不是凸性的，返回True或False
'''

k1 = cv2.isContourConvex(cnt)
print(k1)
k2 = cv2.isContourConvex(cnt)
print(k2)

'''
边界矩形
    直边界矩形：一个直矩形，不考虑对象是否旋转，使用函数cv2.boundingRect()查找得到，(x,y)为矩形左上角坐标，(w,h)为矩形的宽高
    旋转边界矩形：面积是最小的，函数cv2.minAreaRect()，返回的是一个Box2D结构，包含左上角坐标，宽和高以及旋转角度，绘制这个矩形需要4个点，通过函数cv2.boxPoints()获取

'''

x,y,w,h = cv2.boundingRect(cnt)#获取坐标
img11 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
print('x:%d y:%d w:%d h:%d' % (x,y,w,h))

x,y,w,h = cv2.boundingRect(cnt)
img21 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

print(cv2.minAreaRect(cnt))

plt.subplot(121)
plt.imshow(img11)

plt.subplot(122)
plt.imshow(img21)

plt.show()


'''
最小外接圆
函数：cv2.minEnclosingCircle()外切圆
'''

(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),2)
print('x:%d y:%d radius:%d' % (x,y,radius))

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
椭圆拟合
函数：cv2.ellipse()返回值时旋转边界矩形的内切圆
'''

ellipse = cv2.fitEllipse(cnt)
im = cv2.ellipse(img,ellipse,(0,255,0),2)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
直线拟合
'''
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
img = cv2.line(img,(cols - 1,righty),(0,lefty),(0,255,0),2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()