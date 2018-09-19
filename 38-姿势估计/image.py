'''
学习使用calib3D模块在图像中创建3D效果
'''

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


'''
对畸变的图像进行标注
'''

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





with np.load('B.npz') as X:
    mtx,dist,_,_ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


def draw(img,corners,imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img,corner,tuple(imgpts[0].ravel()),(255,0,0),5)
    img = cv2.line(img,corner,tuple(imgpts[1].ravel()),(0,255,0),5)
    img = cv2.line(img,corner,tuple(imgpts[2].ravel()),(0,0,255),5)
    return img


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER,30,0.001)
objp = np.zeros((96*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3)

for fname in glob.glob('*.png'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(7,6),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        rvecs,tvecs,inliers = cv2.solvePnPRansac(objp,corners2,mtx,dist)

        imgpts,jac = cv2.projectPoints(axis,rvecs,tvecs,mtx,dist)
        img =draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0)&0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png',img)


cv2.destroyAllWindows()