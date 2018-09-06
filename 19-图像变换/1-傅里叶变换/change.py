'''
傅里叶变换
函数： cv2.dft()
      cv2.idft()
      傅里叶变换用来分析不同滤波器的频率特性。
      快速傅里叶变换FFT
'''

'''
numpy中的傅里叶变换
np.fft.fft2()可以对信号进行频率转换，输出结果是一个复杂的数组
参数1 输入图像，灰度格式
参数2 可选的，决定输出数组的大小，输出数组大小和输入图像大小一样，如果输出结果比输入图像大，输入图像就要在进行FFT前补0。如果输出结果比输入图像小的话，输入图像就会被切割。
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('flat.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#构建振幅图
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Input Image'),plt.xlabel([]),plt.ylabel([])
plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
plt.show()

rows,cols = img.shape
crow,ccol = rows // 2,cols // 2
fshift[crow - 30:crow + 30,ccol - 30:ccol + 30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
#取绝对值
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(img,cmap='gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])

plt.subplot(132),plt.imshow(img_back,cmap='gray')
plt.title('Image after HPF'),plt.xticks([]),plt.yticks([])

plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'),plt.xticks([]),plt.yticks([])
plt.show()