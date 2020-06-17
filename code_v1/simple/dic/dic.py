import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from get_frames import get_frames

def get_next_ROI(ROI_x_min, ROI_x_max, ROI_y_min, ROI_y_max, current_i_frame, results, width, height, template_width, template_height):
	import numpy as np
	ROI_size_half = 0.06
	curr = results[current_i_frame]
	if current_i_frame > 0:
		prev = results[current_i_frame-1]
		speed_x = curr[0] - prev[0]
		speed_y = curr[1] - prev[1]
	else:
		speed_x = 0
		speed_y = 0
	
	ROI_x_min = np.clip(int(curr[0] - width*ROI_size_half + template_width/2 + speed_x), 0, width)
	ROI_x_max = np.clip(int(curr[0] + width*ROI_size_half + template_width/2 + speed_x), 0, width)
	ROI_y_min = np.clip(int(curr[1] - height*ROI_size_half + template_height/2 + speed_y), 0, height)
	ROI_y_max = np.clip(int(curr[1] + height*ROI_size_half + template_height/2 + speed_y), 0, height)
	return ROI_x_min, ROI_x_max, ROI_y_min, ROI_y_max


def main():
	template = cv2.imread('../clip008-ball_cue.bmp')
	template_height = template.shape[0]
	template_width = template.shape[1]
	template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
	input_fn = '../clip008.mp4'
	output_fn = 'clip008_dic.mp4'
	if input_fn == output_fn:
		raise ValueError('Error: input_fn == output_fn')
	frames, width, height, fps = get_frames(input_fn)
	if len(frames) == 0:
		print('Error: len(frames) == 0')
		exit()
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_fn, fourcc, fps, (width,height))
	# updating these parameters: ROI, results, i_frame
	ROI_x_min = 0
	ROI_x_max = width
	ROI_y_min = 0
	ROI_y_max = height
	results = [(-1,-1) for i in range(len(frames))]
	temp_template = template_hsv	
	for i_frame in range(len(frames)):
		if i_frame % 50 == 0:
			print('Progress: %.2f%%(%d/%d)' % (100*i_frame/len(frames), i_frame, len(frames)))
		frame = frames[i_frame]
		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		roi_frame_hsv = frame_hsv[ROI_y_min:ROI_y_max,ROI_x_min:ROI_x_max]
		res = cv2.matchTemplate(roi_frame_hsv, temp_template, cv2.TM_SQDIFF)
		res = res + 1.50 * cv2.matchTemplate(roi_frame_hsv, template_hsv, cv2.TM_SQDIFF)
		ind = np.unravel_index(np.argmin(res, axis=None), res.shape)
		ind = (ind[0]+ROI_y_min, ind[1]+ROI_x_min)
		results[i_frame] = (ind[1], ind[0])
		# plot object
		rec_start = (ind[1], ind[0])
		rec_end = (ind[1] + template.shape[1] - 1, ind[0] + template.shape[0] - 1)
		cv2.rectangle(frame, rec_start, rec_end, (0,0,255))
		# plot region of interest
		rec_start = (ROI_x_min, ROI_y_min)
		rec_end = (ROI_x_max, ROI_y_max)
		text_loc = (ROI_x_min+8, ROI_y_min+15)
		cv2.rectangle(frame, rec_start, rec_end, (0,255,0), 1)
		cv2.putText(frame, 'region of interest', text_loc, cv2.FONT_HERSHEY_SIMPLEX,
						0.6, (0,255,0), 1, cv2.LINE_AA)
		out.write(frame)
		# update ROI
		ROI_x_min, ROI_x_max, ROI_y_min, ROI_y_max = get_next_ROI(ROI_x_min, ROI_x_max, ROI_y_min, ROI_y_max, i_frame, results, width, height, template_width, template_height)
		# update template
		temp_template = frame_hsv[ind[0]:ind[0]+template_height, ind[1]:ind[1]+template_width]
		#cv2.imshow('new_template', frame[ind[0]:ind[0]+template_height, ind[1]:ind[1]+template_width])
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		
		
	out.release()
	


main()
	
