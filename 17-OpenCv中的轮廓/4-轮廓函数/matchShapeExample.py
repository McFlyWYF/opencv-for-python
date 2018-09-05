import numpy as np
import cv2

img1 = cv2.imread('1.png',0)
img11 = cv2.imread('11.png',0)
img4 = cv2.imread('4.png',0)
img44 = cv2.imread('44.png',0)
img9 = cv2.imread('9.png',0)
img99 = cv2.imread('99.png',0)

ret1,thresh1 = cv2.threshold(img1,127,255,0)
ret11,thresh11 = cv2.threshold(img11,127,255,0)
ret4,thresh4 = cv2.threshold(img4,127,255,0)
ret44,thresh44 = cv2.threshold(img44,127,255,0)
ret9,thresh9  = cv2.threshold(img9,127,255,0)
ret99,thresh99 = cv2.threshold(img99,127,255,0)

image1,contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
cnt1 = contours1[0]
image11,contours11,hierarchy11 = cv2.findContours(thresh11,2,1)
cnt11 = contours11[0]
image4,contours4,hierarchy4 = cv2.findContours(thresh4,2,1)
cnt4 = contours4[0]
image44,contours44,hierarchy44 = cv2.findContours(thresh44,2,1)
cnt44 = contours44[0]
image9,contours9,hierarchy9 = cv2.findContours(thresh9,2,1)
cnt9 = contours9[0]
image99,contours99,hierarchy99 = cv2.findContours(thresh99,2,1)
cnt99 = contours99[0]

ret1 = cv2.matchShapes(cnt1,cnt11,1,0.0)
ret4 = cv2.matchShapes(cnt4,cnt44,1,0.0)
ret9 = cv2.matchShapes(cnt9,cnt99,1,0.0)

print('1的匹配结果：',ret1)
print('4的匹配结果：',ret4)
print('9的匹配结果：',ret9)