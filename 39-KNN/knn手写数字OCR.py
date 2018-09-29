import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#split image to 5000 cells,each 20x20
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

#make it into a numpy
x = np.array(cells)

#prepare train data and test data
train = x[:,:50].reshape(-1,400).astype(np.float32)#size =  (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32)#size =  (2500,400)

k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

#initiate KNN
knn = cv2.ml_KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0 / result.size
print(accuracy)
