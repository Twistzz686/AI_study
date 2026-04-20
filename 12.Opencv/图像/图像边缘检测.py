import cv2

image_np =cv2.imread('./picture.png')

image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

image_blur = cv2.GaussianBlur(image_np,(5,5),1.5)

image_canny = cv2.Canny(image_blur,30,70)

cv2.imshow('image_np',image_np)
cv2.imshow('image_canny',image_canny)
cv2.waitKey(0)