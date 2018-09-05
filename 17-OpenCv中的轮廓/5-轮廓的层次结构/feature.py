'''
轮廓之间的父子关系
'''

'''
轮廓检索模式
    RETR_LIST:提取所有的轮廓，属于同一级组织轮廓
    RETR_EXTERNAL:返回最外边的轮廓，所有的子轮廓会被忽略
    RETR_CCOMP:返回所有的轮廓并将轮廓分为两级组织结构
    RETR_TREE:返回所有轮廓并创建一个完整的组织结构列表
'''

import cv2

img = cv2.imread('test.png',0)

ret,thresh = cv2.threshold(img,127,255,0)
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy)
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy)
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy)
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy)