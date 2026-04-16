import cv2

image_np =cv2.imread('./lena.png')

image_flip = cv2.flip(image_np,0)

cv2.imshow('image_np',image_np)
cv2.imshow('image_flip',image_flip)
cv2.waitKey(0)
