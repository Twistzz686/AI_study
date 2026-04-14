import cv2
import numpy as np

image_np = cv2.imread('./flower.png')

image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

# image_shape = image_gray.shape
# image_thresh = np.zeros((image_shape[0],image_shape[1]),dtype = np.uint8)

thresh = 127
maxval = 255

ret,image_thresh = cv2.threshold(image_gray,thresh,maxval,cv2.THRESH_BINARY_INV)

# for i in range(image_shape[0]):
#     for j in range(image_shape[1]):
#         if image_gray[i,j] > thresh:
#             image_thresh[i,j] = 0
#         else:
#             image_thresh[i,j] = maxval

# cv2.imshow('image',image_gray)
# cv2.imshow('image_thresh',image_thresh)
cv2.imshow('image_thresh',image_thresh)
cv2.waitKey(0)