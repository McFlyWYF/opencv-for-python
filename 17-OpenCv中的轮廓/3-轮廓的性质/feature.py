'''
长宽比
边界矩形的长宽比： 宽w / 长h
'''

import cv2
import numpy as np

img = cv2.imread('test.png',0)
ret,thresh = cv2.threshold(img,127,255,0)
image,contours,hierarchy = cv2.findContours(thresh,1,2)
cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w) / h
print('w:%d h:%d' % (w,h))
print('长宽比：',aspect_ratio)


'''
轮廓面积与边界矩形面积的比
'''

area = cv2.contourArea(cnt)#轮廓面积
x,y,w,h = cv2.boundingRect(cnt)
rect_area = w * h#边界矩形面积
extent = float(area) / rect_area
print("area:%d  rect_area:%d" % (area,rect_area))
print('轮廓面积与边界矩形面积的比',extent)


'''
轮廓面积与凸包面积的比
'''
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)

solidity = float(area) / hull_area

print('area:%d  hull_area:%d' % (area,hull_area))
print('轮廓面积与凸包面积的比',solidity)


'''
与轮廓面积相等的圆形的直径：根号((4 x 轮廓面积) / π)
'''

equi_diameter = np.sqrt(4 * area / np.pi)
print('直径为：',equi_diameter)


'''
方向：对象的方向，返回长轴和短轴的长度
'''

(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
print(angle)


'''
掩模和像素点
'''
img1 = cv2.imread('test.png')
imgray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
print(pixelpoints)

'''
最大值和最小值及他们的位置
'''
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(imgray,mask=mask)
print(min_val,max_val,min_loc,max_loc)
cv2.circle(img, min_loc, 5, [0, 0, 255], -1)
cv2.circle(img, max_loc, 5, [0, 0, 255], -1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
平均颜色及平均度
'''

mean_val = cv2.mean(img,mask=mask)
print(mean_val)


'''
极点，最上面，最下面，最左边，最右边的点
'''
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
print(leftmost,rightmost,topmost,bottommost)


img2 = cv2.imread('test.png')
cv2.circle(img2, leftmost, 5, [255, 0, 0], -1)
cv2.circle(img2, rightmost, 5, [0, 255, 0], -1)
cv2.circle(img2, topmost, 5, [0, 0, 255], -1)
cv2.circle(img2, bottommost, 5, [255, 0, 0], -1)

cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

