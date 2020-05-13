import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from get_frames import get_frames
from imshow_scaling import imshow_scaling


def get_mask_by_background_subtraction(frame_orig, background_orig, diffthres=25):
	#must be 3-channel images
	frame = cv2.GaussianBlur(frame_orig, (5,5), 0) # blur
	frame = frame.astype('int')
	background = cv2.GaussianBlur(background_orig, (5,5),0)
	background = background.astype('int')
	subtracted = np.subtract(frame, background)
	subtracted[abs(subtracted) <= diffthres] = 0
	subtracted[abs(subtracted) > diffthres] = 1
	subtracted = subtracted.astype('uint8')
	mask = cv2.bitwise_or(cv2.bitwise_or(subtracted[:,:,0], subtracted[:,:,1]), subtracted[:,:,2])
	mask = cv2.erode(mask, np.ones((3,3),np.uint8), iterations=1)
	mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=2)
	mask = cv2.erode(mask, np.ones((3,3),np.uint8), iterations=2)
	mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=2)
	return mask


def get_component_images(frame_orig, mask, min_mask_size=25, max_mask_size=28900):
	# if min_mask_size < 0 then don't filter by min_mask_size
	# if max_mask_size < 0 then don't filter by max_mask_size
	n_labels, labels = cv2.connectedComponents(mask, connectivity=4)
	labels = np.array(labels).astype('int')
	tmp_component_masks = [np.zeros_like(mask) for i in range(n_labels)]
	tmp_component_images = [np.zeros_like(frame_orig) for i in range(n_labels)]
	for y in range(mask.shape[0]):
		for x in range(mask.shape[1]):
			label = labels[y,x]
			tmp_component_masks[label][y,x] = 1
			tmp_component_images[label][y,x,0] = frame_orig[y,x,0]
			tmp_component_images[label][y,x,1] = frame_orig[y,x,1]
			tmp_component_images[label][y,x,2] = frame_orig[y,x,2]
	min_xs = [mask.shape[1] for i in range(n_labels)]
	min_ys = [mask.shape[0] for i in range(n_labels)]
	max_xs = [0 for i in range(n_labels)]
	max_ys = [0 for i in range(n_labels)]
	for y in range(mask.shape[0]):
		for x in range(mask.shape[1]):
			label = labels[y,x]
			if x < min_xs[label]:
				min_xs[label] = x
			if y < min_ys[label]:
				min_ys[label] = y
			if x > max_xs[label]:
				max_xs[label] = x
			if y > max_ys[label]:
				max_ys[label] = y
	tmp_masks = [0 for i in range(n_labels)]
	tmp_images = [0 for i in range(n_labels)]
	for i in range(n_labels):
		tmp_masks[i] = tmp_component_masks[i][min_ys[i]:max_ys[i]+1, min_xs[i]:max_xs[i]+1].copy()
		tmp_images[i] = frame_orig[min_ys[i]:max_ys[i]+1, min_xs[i]:max_xs[i]+1].copy()
	mask_sizes = [np.sum(tmp_component_masks[i]) for i in range(n_labels)]
	good_labels = [i for i in range(n_labels)]
	if min_mask_size >= 0:
		good_labels = [label for label in good_labels if mask_sizes[label] >= min_mask_size]
	if max_mask_size >= 0:
		good_labels = [label for label in good_labels if mask_sizes[label] <= max_mask_size]
	ret_masks = [tmp_masks[label] for label in good_labels]
	ret_images = [tmp_images[label] for label in good_labels]
	return len(good_labels), ret_masks, ret_images


def create_component_images(frames, background_orig, output_prefix):
	fn_masks = []
	fn_images = []
	for i in range(len(frames)):
		if i % 10 == 0:
			print('ball_extraction, create_component_images, progress: %.2f%%(%d/%d)' % (100*i/len(frames), i, len(frames)))
		mask = get_mask_by_background_subtraction(frames[i], background_orig)
		n_labels, component_masks, component_images = get_component_images(frames[i], mask, min_mask_size=64, max_mask_size=1600)
		ball_count = 0
		for i_label in range(n_labels):
			t_c_mask = component_masks[i_label]
			height, width = t_c_mask.shape[0], t_c_mask.shape[1]
			if height > width * 2 or width > height * 2:
				continue
			diff = calculate_circle_difference(t_c_mask)
			if diff < 0.095:
				fn_mask = '%smask_%d_%d.bmp' % (output_prefix, i, ball_count)
				fn_image = '%simage_%d_%d.bmp' % (output_prefix, i, ball_count)
				fn_masks.append(fn_mask)
				fn_images.append(fn_image)
				ball_count += 1
				cv2.imwrite(fn_mask, component_masks[i_label]*255)
				cv2.imwrite(fn_image, component_images[i_label])
	return fn_masks, fn_images
	

def create_circle_mask(img, r=1.23):
	height, width = img.shape[0], img.shape[1]
	circle_mask = np.zeros([height, width]).astype('int')
	cy, cx = (height-1)/2, (width-1)/2
	for y in range(height):
		for x in range(width):
			dy = (y-cy)/cy
			dx = (x-cx)/cx
			if dy*dy+dx*dx <= r:
				circle_mask[y,x] = 1
	return circle_mask
	
	
def calculate_circle_difference(img):
	circle_img = create_circle_mask(img)
	diff_img = np.abs(img.astype('int') - circle_img.astype('int'))
	diff_cnt = np.sum(diff_img)
	num_px = img.shape[0] * img.shape[1]
	return float(diff_cnt/num_px)


def ball_extraction(input_fn, background_fn, tmp_dir):
	frames, width, height, fps = get_frames(input_fn)
	if len(frames) == 0:
		print('Error: len(frames) == 0')
		exit()
	background_orig = cv2.imread(background_fn)
	fn_masks, fn_images = create_component_images(frames, background_orig, tmp_dir)
	return fn_masks, fn_images
	