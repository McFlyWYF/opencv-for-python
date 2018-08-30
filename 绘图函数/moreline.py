'''
多边形
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
需要指出每个顶点的坐标，用点的坐标构建一个大小等于行数X1X2的数组，
行数就是点的数目，数组数据类型是int32
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

a = np.array([[[10,10], [100,10], [120,100], [10,120]]], dtype = np.int32)
print(a.shape)

im = np.zeros([240, 320], dtype = np.uint8)
cv2.polylines(im, a, True, 255)#第三个参数是False的话，多边形是不闭合的
# plt.imshow(im)
# plt.show()
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()