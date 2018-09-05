'''
1.凸缺陷，找凸缺陷
2.找某一点到一个多边形的最短距离
3.不同形状的匹配
'''


import cv2
import numpy as np

img = cv2.imread('test.png',0)
ret,thresh = cv2.threshold(img,127,255,0)
image,contours,hierarchy = cv2.findContours(thresh,1,2)
cnt = contours[0]


'''
凸缺陷
函数：cv2.convexityDefect()
'''

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)#起始点，结束点，距离轮廓凸包最远点，最远点到轮廓凸包的距离,返回的前三个值是索引
print(defects)



img1 = cv2.imread('test2.png')

img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(img_gray,127,255,0)
image,contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
cnt1 = contours1[0]

hull1 = cv2.convexHull(cnt1,returnPoints=False)
defects1 = cv2.convexityDefects(cnt1,hull1)
for i in range(defects1.shape[0]):
    s,e,f,d = defects1[i,0]
    start = tuple(cnt1[s][0])
    end = tuple(cnt1[e][0])
    far = tuple(cnt1[f][0])
    cv2.line(img1,start,end,[0,255,0],2)
    cv2.circle(img1,far,5,[255,0,0],-1)

cv2.imshow('img',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('shortcoming.png',img1)


'''
求解图像中一个点到一个对象轮廓的最短距离，点在轮廓外部，返回值为负，在轮廓上，返回值为0，在轮廓内部，返回值为正。
'''

dist = cv2.pointPolygonTest(cnt,(50,50),True)#设置为True，会计算距离，如果是False,只会判断这个点与轮廓之间的位置关系。（返回值为+1，-1,0）
print(dist)
dist1 = cv2.pointPolygonTest(cnt,(100,100),False)
print(dist1)

'''
形状匹配
函数：cv2.matchShape()比较两个形状或轮廓的相似度，如果返回值越小，匹配越好
'''

img2 = cv2.imread('A.png',0)
img3 = cv2.imread('B.png',0)
img4 = cv2.imread('C.png',0)

ret,thresh2 = cv2.threshold(img2,127,255,0)
ret,thresh3 = cv2.threshold(img3,127,255,0)
ret,thresh4 = cv2.threshold(img4,127,255,0)

image,contours2,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours2[0]
image,contours3,hierarchy = cv2.findContours(thresh3,2,1)
cnt3 = contours3[0]
image,contours4,hierarchy = cv2.findContours(thresh4,2,1)
cnt4 = contours4[0]

ret1 = cv2.matchShapes(cnt2,cnt3,1,0.0)
ret2 = cv2.matchShapes(cnt2,cnt4,1,0.0)
ret3 = cv2.matchShapes(cnt3,cnt4,1,0.0)

print('A和B的匹配：',ret1)
print('A和C的匹配：',ret2)
print('B和C的匹配：',ret3)