import cv2

image_np = cv2.imread('./lena.png')

image_shape = image_np.shape

angle = 45
scale = 0.5
M = cv2.getRotationMatrix2D((image_shape[1]/2,image_shape[0]/2),angle,scale)

image_rotation = cv2.warpAffine(image_np,M,(image_shape[1],image_shape[0]),flags = cv2.INTER_LINEAR,borderMode = cv2.BORDER_CONSTANT,borderValue=(0,0,0))
cv2.imshow('image_np',image_np)
cv2.imshow('image_rotation',image_rotation)
cv2.waitKey(0)
