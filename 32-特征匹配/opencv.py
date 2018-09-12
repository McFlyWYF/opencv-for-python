'''
特征匹配
蛮力匹配和FLANN匹配
蛮力匹配：在第一幅图选取一个关键点依次与第二幅图的每个关键点进行距离测试，最后返回距离最近的关键点。
'''

#对ORB描述符进行蛮力匹配
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('os.jpg',0)
img2 = cv2.imread('os_part.jpg',0)

#initiate SIFT detector
orb = cv2.ORB_create()

#find the keypoints and descriptors with SOFT
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

#create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)

#Match descriptors
matches = bf.match(des1,des2)#返回值是一个DMatch对象列表

#sort them in the order of their distance
matches = sorted(matches,key=lambda x : x.distance)
print(matches)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],None,flags=2)
plt.imshow(img3)
plt.show()


#对SIFT描述符进行蛮力匹配和比值测试，使用BFMatcher.knnMatch()获得K对最佳匹配
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('os.jpg',0)
img2 = cv2.imread('os_part.jpg',0)

#initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

#find the keypoints
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

#BFMatcher with default params
bf = cv2.BFMatcher()
matches - bf.knnMatch(des1,des2,k = 2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],None,flags=2)
plt.imshow(img3)
plt.show()



