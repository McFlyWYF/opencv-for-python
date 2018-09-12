'''
ORB是FAST关键点检测和BRIEF关键点描述器的结合体，使用FAST找打关键点，使用Harris角点检测对这些关键点进行排序找到其中的前N个点。
ORB使用的是BRIEF描述符。
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('shape.png',0)

#initiate STAR detector
orb = cv2.ORB_create()

#find the keypoints with ORB
kp = orb.detect(img,None)

#compute the descriptors with ORB
kp,des = orb.compute(img,kp)

#draw only keypoints location
img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0),flags=0)
plt.imshow(img2)
plt.show()
