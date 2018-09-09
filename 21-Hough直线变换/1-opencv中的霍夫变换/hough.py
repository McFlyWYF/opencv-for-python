'''
函数：cv2.HoughLines()返回值就是(ρ,θ)，ρ的单位是像素，θ的单位是弧度。
参数1：二值化图像
参数2：ρ的精确度
参数3：θ的精确度
参数4：阈值，只有累加其中的值高于阈值时才被认为是条直线
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('dave.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi / 180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)

plt.imshow(img)
plt.show()