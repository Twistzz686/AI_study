import cv2

image_np = cv2.imread('./tubao.png')

image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

ret,image_thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)

contours,hierarchy = cv2.findContours(image_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
hull = cv2.convexHull(cnt)

image_poly = cv2.polylines(image_np,[hull],True,(0,0,255))

cv2.imshow('image_np',image_np)
cv2.imshow('image_poly',image_poly)
cv2.waitKey(0)