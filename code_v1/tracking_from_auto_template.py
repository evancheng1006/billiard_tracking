# new tracking using background subtraction (mask) and multiple templates

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import re
import random


from get_frames import get_frames
from imshow_scaling import imshow_scaling
	

def read_templates(template_dir):
	# example template filename: 'auto-template-image_0_1084.jpg'
	# templates: dict of lists(templates[label] is a list of numpy array images)
	import re
	all_fns = [fn for fn in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, fn))]
	tmp_fns = [fn for fn in all_fns if len(re.findall(r"image_\d+_\d+$",os.path.splitext(fn)[0])) > 0]
	tmp_fns = [(re.findall(r"image_\d+_\d+$", os.path.splitext(fn)[0])[0], fn) for fn in tmp_fns]
	tmp_fns = [(int(matched.split('_')[1]), fn) for matched, fn in tmp_fns]
	labels = sorted(list(set([label for label, fn in tmp_fns])))
	templates = {label: [] for label in labels}
	for i in range(len(tmp_fns)):
		if i % 500 == 0:
			print('read template progress: %.2f%%(%d/%d)' % (100*i/len(tmp_fns), i, len(tmp_fns)))
		label, fn = tmp_fns[i]
		img = cv2.imread(os.path.join(template_dir, fn))
		templates[label].append(img)
	return templates
	
	
def get_max_template_shape(templates):
	# templates: dict of lists(templates[label] is a list of numpy array images)
	max_height = 0
	max_width = 0
	for label in templates:
		for img in templates[label]:
			height = img.shape[0]
			width  = img.shape[1]
			if height > max_height:
				max_height = height
			if width > max_width:
				max_width = width
	return (max_height, max_width)


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


def get_average_sqdiff(frame_orig, template_orig, x_offset, y_offset):
	if (y_offset+template_orig.shape[0]) > frame_orig.shape[0]:
		return np.Infinity
	if (x_offset+template_orig.shape[1]) > frame_orig.shape[1]:
		return np.Infinity
	part_frame = frame_orig[y_offset:y_offset+template_orig.shape[0],x_offset:x_offset+template_orig.shape[1]].astype('int')
	template = template_orig.astype('int')
	average_sqdiff = np.mean(np.square(np.subtract(part_frame,template)))
	return average_sqdiff

	
def get_masks(frames, background_orig, mask_dilate_kernel):
	masks = [-1 for i in range(len(frames))]
	for i in range(len(frames)):
		if i % 500 == 0:
			print('mask calcultion progress: %.2f%%(%d/%d)' % (100*i/len(frames), i, len(frames)))
		masks[i] = get_mask_by_background_subtraction(frames[i], background_orig, 25)
		masks[i] = cv2.dilate(masks[i], mask_dilate_kernel, iterations=1)	
	return masks


def tracking_from_auto_template(input_fn, background_fn, template_dir, output_tracking_result_fn,
			max_template_sample_size=1, avg_sqdiff_thres = 700, blur_kernel_size = 3):
	fo = open(output_tracking_result_fn, "w")
	fo.write("frame_num,detection_result\n")
	fo.flush()
	
	templates = read_templates(template_dir)
	max_height, max_width = get_max_template_shape(templates)
	mask_dilate_kernel = np.ones([int(max_height*0.4), int(max_width*0.4)],np.uint8)
	
	background_orig = cv2.imread(background_fn)
	frames, width, height, fps = get_frames(input_fn)
	
	if len(frames) == 0:
		print('Error: no frame in video')
		exit()
	
	masks = get_masks(frames, background_orig, mask_dilate_kernel)
	#max_template_sample_size = 1 # usually 1, otherwise too slow
	#avg_sqdiff_thres = 700 # depends on the noise ant gaussian blur parameters, can't be too small
	#blur_kernel_size = 3
	
	sample_sizes = {label : min(len(templates[label]),max_template_sample_size) for label in templates}
	detection_results = [{} for i in range(len(frames))]
	# calculate differences
	for i in range(len(frames)):
		#if i % 1 == 0:
		#	print('difference calcultion progress: %.2f%%(%d/%d)' % (100*i/len(frames), i, len(frames)))
		blurred = cv2.GaussianBlur(frames[i], (blur_kernel_size,blur_kernel_size),0)
		blurred = blurred.astype('int')
		#print('Mask size:', np.sum(masks[i]))
		#cv2.imshow('mask', masks[i]*255)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		for label in templates:
			dists = np.Infinity * np.ones([height, width])
			best_matched_heights = np.zeros([height, width]).astype('int')
			best_matched_widths = np.zeros([height, width]).astype('int')
			sameple_size = sample_sizes[label]
			for y in range(height):
				for x in range(width):
					if masks[i][y,x] == 0:
						continue
					# because there are too many images in a template, to speed up, we randomly sample a few templates.
					samples = random.sample(templates[label], sameple_size)
					for img in samples:
						tmp_dist = get_average_sqdiff(blurred, img, x, y)
						if tmp_dist < dists[y,x]:
							dists[y,x] = tmp_dist
							best_matched_heights[y,x] = int(img.shape[0])
							best_matched_widths[y,x] = int(img.shape[1])
			ind = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
			#print(i, label, ind, dists[ind])
			if dists[ind] < avg_sqdiff_thres: # is a match
				best_matched_size = (best_matched_heights[ind], best_matched_widths[ind])
				detection_results[i][label] = (ind, best_matched_size)
		result_str = str(detection_results[i]).replace('"','\'').replace(' ','')
		result_str = '%d,"%s"\n' % (i, result_str)
		fo.write(result_str)
		fo.flush()
		print(result_str)
	fo.close()
	return
			

				