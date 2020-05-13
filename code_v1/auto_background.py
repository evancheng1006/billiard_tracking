# https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_frames import get_frames


def histogram_smooth(hist256, filter_diff=2):
	# smoothing using max filter + mean filter
	window_mins = [max(i-filter_diff, 0) for i in range(256)]
	window_maxs = [min(i+filter_diff, 255) for i in range(256)]
	# max filter
	hist256_max = [np.max(hist256[window_mins[i]:window_maxs[i]+1]) for i in range(256)]
	# mean filter
	hist256_max_mean = [np.mean(hist256_max[window_mins[i]:window_maxs[i]+1]) for i in range(256)]
	return hist256_max_mean


def uint8_array_to_hist(arr):
	arr_hist = [0 for i in range(256)]
	for i in range(len(arr)):
		arr_hist[arr[i]] += 1
	return arr_hist


def uint8_array_to_hist_weighted(arr, masks_px, masked_weight):
	arr_hist = [0.0 for i in range(256)]
	for i in range(len(arr)):
		if masks_px[i] == 0:
			arr_hist[arr[i]] += 1
		else:
			arr_hist[arr[i]] += masked_weight
	return arr_hist


def get_best_color_from_hist(bs, gs, rs, b_hist, g_hist, r_hist):
	# best color: the color that has the highest frequency score
	frequency_score = [b_hist[bs[i]]+g_hist[gs[i]]+r_hist[rs[i]] for i in range(len(bs))]
	best_idx = np.argmax(np.asarray(frequency_score))
	best_color_bgr = (bs[best_idx], gs[best_idx], rs[best_idx])
	return best_color_bgr


def get_best_color(bs, gs, rs, masks_px, masked_weight=0.012):
	weighted_b_hist = uint8_array_to_hist_weighted(bs, masks_px, masked_weight)
	weighted_g_hist = uint8_array_to_hist_weighted(gs, masks_px, masked_weight)
	weighted_r_hist = uint8_array_to_hist_weighted(rs, masks_px, masked_weight)
	smoothed_weighted_b_hist = histogram_smooth(weighted_b_hist)
	smoothed_weighted_g_hist = histogram_smooth(weighted_g_hist)
	smoothed_weighted_r_hist = histogram_smooth(weighted_r_hist)
	ret = get_best_color_from_hist(bs, gs, rs, smoothed_weighted_b_hist, smoothed_weighted_g_hist, smoothed_weighted_r_hist)
	return ret


def get_hough_circle_mask(gray_frame, r_extend=2):
	# purpose of r_extend: hough circle detection could be detect a circle that is too small
	mask = np.zeros([gray_frame.shape[0], gray_frame.shape[1]]).astype('uint8')
	circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1.2, 5, param1=100, param2=13, minRadius=4, maxRadius=10)
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		for (x, y, r) in circles:
			cv2.circle(mask, (x, y), r+r_extend, 1, -1)
	return mask
	
	
def get_masked(frames):
	masks = [0 for i in range(len(frames))]
	for i in range(len(frames)):
		frame = frames[i]
		tmp_mask_b = get_hough_circle_mask(frame[:,:,0])
		tmp_mask_g = get_hough_circle_mask(frame[:,:,1])
		tmp_mask_r = get_hough_circle_mask(frame[:,:,2])
		masks[i] = cv2.bitwise_or(cv2.bitwise_or(tmp_mask_b, tmp_mask_g), tmp_mask_r)
	return masks


def auto_background(input_fn, output_fn):
	frames, width, height, fps = get_frames(input_fn)

	if len(frames) == 0:
		print('Error: len(frames) == 0')
		exit()

	print('Computing Masks...')
	masks = get_masked(frames)


	result = np.zeros_like(frames[0])
	print('Generating Result...')
	#for y in range(150, 180):
	for y in range(height):
		if y % 2 == 0:
			print('auto_background progress: %.2f%%(%d/%d)' % (100*y/height, y, height))
		#for x in range(240, 270):
		for x in range(width):
			bs = [frames[i][y, x, 0] for i in range(len(frames))]
			gs = [frames[i][y, x, 1] for i in range(len(frames))]
			rs = [frames[i][y, x, 2] for i in range(len(frames))]
			masks_px = [masks[i][y, x] for i in range(len(frames))]

			bgr_px = get_best_color(bs, gs, rs, masks_px)
			result[y, x, 0] = bgr_px[0]
			result[y, x, 1] = bgr_px[1]
			result[y, x, 2] = bgr_px[2]

	cv2.imwrite(output_fn, result)
	return
