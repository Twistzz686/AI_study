import cv2

image_np = cv2.imread('./31.png')

image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

ret,image_binary = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours,hierarchy = cv2.findContours(image_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image_np,contours,-1,(0,0,255))

cv2.imshow('image_binary',image_binary)
cv2.imshow('image_np',image_np)
cv2.waitKey(0)