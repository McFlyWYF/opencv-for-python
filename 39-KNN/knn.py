import cv2
import numpy as np
import matplotlib.pyplot as plt

#feature
trainData = np.random.randint(1,100,(25,2)).astype(np.float32)

#label
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

red = trainData[responses.ravel() == 0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

plt.show()

newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv2.ml_KNearest()
knn.train(trainData,responses)
ret,results,neighbours,dist = knn.findNearest(newcomer,3)

print('resullt ',results,"\n")
print('neighbours: ',neighbours,"\n")
print("distance ",dist)
plt.show()