'''
颜色空间转换
    从BGR到灰度图，cv2.COLOR_BGR2GRAY
    从BGR到HSV，cv2.COLOR_BGR2HSV

    函数
        cv2.cvtColor(),第一个参数，图片，第二个参数，转换类型
        cv2.inRange()
'''

'''
颜色空间转换
'''

import cv2
import numpy as np

img1 = cv2.imread('hourse.jpg')
res = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
cv2.imshow('image',res)
cv2.imwrite('hourse_gray.png',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


#通道拆分
b,g,r = cv2.split(img1)
cv2.imshow('blue',b)
cv2.imwrite('blue.png',b)
cv2.waitKey(0)
cv2.destroyAllWindows()

#通道合并
zeros = np.zeros(img1.shape[:2],res.dtype)
res = cv2.merge((b,zeros,zeros))
cv2.imshow('blue',res)
cv2.imwrite('merge.png',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

#图像扩边
BLUE = [255,0,0]
df = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
cv2.imshow('blue',df)
cv2.imwrite('expand.png',df)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
物体跟踪

    从视频获取每一帧图像
    将图像转换到HSV空间
    设置HSV阈值到蓝色范围
    获取蓝色物体
'''

cap = cv2.VideoCapture(0)

while(1):

    #获取每一帧
    ret,frame = cap.read()

    #转换到hsv
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #设置蓝色的阈值
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])


    #根据阈值构建掩模
    mask = cv2.inRange(hsv,lower_blue,upper_blue)

    #对原图像和掩模进行位运算
    res = cv2.bitwise_and(frame,frame,mask = mask)

    #显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) &0xFF

    if k == 27:
        break

#关闭窗口
cv2.destroyAllWindows()


'''
怎样找到要跟踪对象的HSV值
'''
#找到绿色的HSV的值
green = np.uint8([[[0,255,0]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print(hsv_green)