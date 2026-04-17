import cv2
import numpy as np

image_np = cv2.imread('./shudu.png')


# filter2D
# kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
# image_filter2D = cv2.filter2D(image_np,-1,kernel)
#
# cv2.imshow('image_np',image_np)
# cv2.imshow('image_filter2D',image_filter2D)
# cv2.waitKey(0)



# sobel
# image_Sobel = cv2.Sobel(image_np,-1,1,0)
#
# cv2.imshow('image_np',image_np)
# cv2.imshow('image_Sobel',image_Sobel)
# cv2.waitKey(0)

# Laplacian
image_Laplacian = cv2.Laplacian(image_np,-1)

cv2.imshow('image_np',image_np)
cv2.imshow('image_Laplacian',image_Laplacian)
cv2.waitKey(0)