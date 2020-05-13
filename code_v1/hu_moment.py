import cv2
import numpy as np

def hu_moment(img_original, invariants='h'):
	img = img_original.copy().astype('double')
	img_h, img_w = img.shape[0], img.shape[1]
	xx_raw = np.zeros(img.shape)
	yy_raw = np.zeros(img.shape)
	for y in range(img_h):
		for x in range(img_w):
			xx_raw[y, x] = x
			yy_raw[y, x] = y
	
	m00 = np.sum(img, axis=None)
	m10 = np.sum(np.multiply(img,xx_raw), axis=None)
	m01 = np.sum(np.multiply(img,yy_raw), axis=None)
	x_mean = m10 / m00
	y_mean = m01 / m00
	
	xx1 = np.subtract(xx_raw, x_mean*np.ones(img.shape))
	yy1 = np.subtract(yy_raw, y_mean*np.ones(img.shape))
	xx2 = np.multiply(xx1, xx1)
	yy2 = np.multiply(yy1, yy1)
	xx3 = np.multiply(xx2, xx1)
	yy3 = np.multiply(yy2, yy1)
	xx1yy1 = np.multiply(xx1, yy1)
	xx1yy2 = np.multiply(xx1, yy2)
	xx2yy1 = np.multiply(xx2, yy1)
	
	u00 = m00
	u01 = np.sum(np.multiply(img,yy1), axis=None)
	u10 = np.sum(np.multiply(img,xx1), axis=None)
	u02 = np.sum(np.multiply(img,yy2), axis=None)
	u11 = np.sum(np.multiply(img,xx1yy1), axis=None)
	u20 = np.sum(np.multiply(img,xx2), axis=None)
	u03 = np.sum(np.multiply(img,yy3), axis=None)
	u12 = np.sum(np.multiply(img,xx1yy2), axis=None)
	u21 = np.sum(np.multiply(img,xx2yy1), axis=None)
	u30 = np.sum(np.multiply(img,xx3), axis=None)

	n00 = u00 / np.power(u00, 1) # should be one
	n01 = u01 / np.power(u00, 1.5) # should be zero
	n10 = u10 / np.power(u00, 1.5) # should be zero
	n02 = u02 / np.power(u00, 2)
	n11 = u11 / np.power(u00, 2)
	n20 = u20 / np.power(u00, 2)
	n03 = u03 / np.power(u00, 2.5)
	n12 = u12 / np.power(u00, 2.5)
	n21 = u21 / np.power(u00, 2.5)
	n30 = u30 / np.power(u00, 2.5)
	
	if invariants == 'n':
		return {'n02':n02, 'n11':n11, 'n20':n20, 'n03':n03, 'n12':n12, 'n21':n21, 'n30':n30}
	
	h1 = n20 + n02
	h2 = ((n20-n02)**2) + 4*(n11**2)
	h3 = ((n30-3*n12)**2) + ((3*n21-n03)**2)
	h4 = ((n30+n12)**2) + ((n21+n03)**2)
	h5 = (n30-3*n12)*(n30+n12)*(((n30+n12)**2)-3*((n21+n03)**2)) + (3*n21-n03)*(n21+n03)*(3*((n30+n12)**2)-((n21+n03)**2))
	h6 = (n20-n02)*(((n30+n12)**2)-((n21+n03)**2)) + 4*n11*(n30+n12)*(n21+n03)
	h7 = (3*n21-n03)*(n30+n12)*(((n30+n12)**2)-3*((n21+n03)**2)) - (n30-3*n12)*(n21+n03)*(3*((n30+n12)**2)-((n21+n03)**2))
	
	if invariants == 'h':
		return np.array([h1, h2, h3, h4, h5, h6, h7])
	
	return np.array([h1, h2, h3, h4, h5, h6, h7])