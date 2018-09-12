'''
BRIEF是一种特征描述符，直接得到一个二进制字符串，使用的是已经平滑后的图像不提供查找特征的方法。
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('shape.png',0)

#initiate STAR
star = cv2.FeatureDetector_create("STAR")

#initiate BRIEF
brief = cv2.DescriptorMatcher_create('BRIEF')

#查找特征点
kp = star.detect(img,None)

#compute des with BRIEF
kp,des = brief.compute(img,kp)

print(brief.getInt('bytes'))
print(des.shape)