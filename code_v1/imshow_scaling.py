def imshow_scaling(title, img, scale=1):
	import cv2
	height = img.shape[0]
	width = img.shape[1]
	new_size = (int(height*scale), int(width*scale))
	cv2.imshow(title, cv2.resize(img, new_size))
	return