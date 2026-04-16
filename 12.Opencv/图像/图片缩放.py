import cv2

image_np =cv2.imread('./lena.png')

image_resize = cv2.resize(image_np,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)

cv2.imshow('image_resize',image_resize)
cv2.waitKey(0)