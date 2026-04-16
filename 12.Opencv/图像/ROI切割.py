import cv2

image_np =cv2.imread('./lena.png')

x_min,x_max = 150,270
y_min,y_max = 150,290

image_roi = image_np[y_min:y_max,x_min:x_max]


cv2.imshow('image_np',image_np)
cv2.imshow('image_roi',image_roi)
cv2.waitKey(0)