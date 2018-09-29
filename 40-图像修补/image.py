import cv2
import matplotlib.pyplot as plt


'''
创建一个和原图像相同的掩模图像
'''
img = cv2.imread('messi.png')
mask = cv2.imread('mask.png',0)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)


plt.subplot(221)
plt.imshow(img)

plt.subplot(222)
plt.imshow(mask)

plt.subplot(223)
plt.imshow(dst)

plt.show()