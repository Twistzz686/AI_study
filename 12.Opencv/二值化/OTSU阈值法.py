import cv2

image_np = cv2.imread('./flower.png')

image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

thresh = 127
maxval = 255

ret,image_thresh=cv2.threshold(image_gray,thresh,maxval,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('image_thresh',image_thresh)
cv2.waitKey(0)