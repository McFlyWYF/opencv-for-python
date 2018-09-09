'''
使用模板匹配在一幅图像中查找目标
函数：cv2.matchTemplate()
     cv2.minMaxLoc()

原理：模板匹配是用来在一幅大图中搜寻查找模板图像位置的方法。
输入图像的大小是W X H，模板大小是w x h，输出结果就是W - w +1，H - h + 1.
'''

#在图片中搜索梅西的面部

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('messi5.jpg',0)
img2 = img.copy()
template = cv2.imread('messi_face.png',0)
w,h = template.shape[::-1]
print('w：%d,h：%d' % (w,h))

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()

    # eval 语句用来计算存储在字符串中的有效 Python 表达式
    method = eval(meth)

    res = cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc

    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w,top_left[1] + h)

    cv2.rectangle(img,top_left,bottom_right,255,2)

    plt.subplot(121)
    plt.imshow(res,cmap='gray')
    plt.title('Matching Result')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(img,cmap='gray')
    plt.title('Dected Point')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle(meth)
    plt.show()