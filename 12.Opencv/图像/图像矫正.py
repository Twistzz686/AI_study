import cv2
import numpy as np

image_np = cv2.imread('./tang.png')

image_shape = image_np.shape

points1 = np.float32([[148,80],[437,114],[94,247],[423,288]])
points2 = np.float32([[0,0],[image_shape[1],0],[0,image_shape[0]],[image_shape[1],image_shape[0]]])

M = cv2.getPerspectiveTransform(points1,points2)

image_warpPerspective = cv2.warpPerspective(image_np,M,(image_shape[1],image_shape[0]))

cv2.imshow('image_np',image_np)
cv2.imshow('image_warpPerspective',image_warpPerspective)
cv2.waitKey(0)