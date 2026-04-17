import cv2

image_np = cv2.imread('./lena.png')
logo = cv2.imread('./logo.png')

logo_shape = logo.shape

roi = image_np[:logo_shape[0],:logo_shape[1]]

logo_gray = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(logo_gray,127,255,cv2.THRESH_BINARY_INV)

image_and = cv2.bitwise_and(roi,roi,mask = mask)

dst = cv2.add(image_and,logo)

image_np[:logo_shape[0],:logo_shape[1]] = dst

cv2.imshow('image_np',image_np)
cv2.waitKey(0)