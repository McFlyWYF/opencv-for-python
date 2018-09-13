'''
联合使用特征提取和calib3d模块的findHomography在复杂图像中查找已知对象
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


MIN_MATCH_COUNT = 10

img1 = cv2.imread('os.jpg',0)
img2 = cv2.imread('os_part.jpg',0)

#initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

#find the keypoints and descriptors with SIFT
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

#store all
good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    #获取关键点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    #M为变换矩阵
    M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    #获取原图像的高和宽
    h,w = img1.shape
    #使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #原图像为灰度图
    cv2.polylines(img2,[np.int32(dst)],True,255,10,cv2.LINE_AA)

else:
    print('Not enough matches are found - %d/%d' % (len(good),MIN_MATCH_COUNT))
    matchesMask = None


#绘制inliers
draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask,flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3,'gray')
plt.show()