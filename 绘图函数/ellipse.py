import cv2
import numpy as np

'''
矩形
'''

img = np.zeros((512,512,3),np.uint8)

#中心点坐标，长轴和短轴长度，沿逆时针方向旋转的角度
cv2.ellipse(img, center=(256, 256), axes=(100, 50), angle=0, startAngle=0, endAngle=180, color=255,
            thickness=-1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()