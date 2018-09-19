'''
对畸变的图像进行标注
'''

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

objp = np.zeros((6*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []#3d points
imgpoints = []#2d points

images = glob.glob('*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,corners = cv2.findChessboardCorners(gray,(7,6),None)

    if ret == True:
        objpoints.append(objp)

        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        img = cv2.drawChessboardCorners(img,(7,6),corners,ret)

        cv2.imshow('img',img)
cv2.waitKey()

cv2.destroyAllWindows()


'''
畸变矫正
'''

ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
img = cv2.imread('fangge.png')
h,w = img.shape[:2]
newcameramtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv2.undistort(img,mtx,dist,None,newcameramtx)

x,y,w,h = roi
dst = dst[y:y+h,x:x+w]

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(dst)
plt.show()



