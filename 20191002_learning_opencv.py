import cv2
import numpy as np

img=cv2.imread(r'D:\LIFE\PICS\ACGF\Frozen_poster.jpg',0)
print(img)
edges=cv2.Canny(img,30,70)
cv2.imshow('canny',np.hstack((img,edges)))
img_edges=np.hstack((img,edges))