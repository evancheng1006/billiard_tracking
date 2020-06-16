import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

template = cv2.imread('../clip008-ball_cue.bmp')
template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
input_fn = '../clip008.mp4'
output_fn = 'clip008_original.mp4'

if input_fn == output_fn:
	raise ValueError('Error: input_fn == output_fn')


cap = cv2.VideoCapture(input_fn)

if cap.isOpened():
	width  = int(cap.get(3))
	height = int(cap.get(4))
	fps = cap.get(5) # float
	frame_count = int(cap.get(7))
else:
	raise ValueError('cannot open video file %s.' % fn)
	
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_fn, fourcc, fps, (width,height))

i_frame = 0
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		i_frame += 1
		if i_frame % 50 == 0:
			print('Progress: %.2f%%(%d/%d)' % (100*i_frame/frame_count, i_frame, frame_count))
		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		res = cv2.matchTemplate(frame_hsv, template_hsv, cv2.TM_SQDIFF)
		ind = np.unravel_index(np.argmin(res, axis=None), res.shape)
		rec_start = (ind[1], ind[0])
		rec_end = (ind[1] + template.shape[1] - 1, ind[0] + template.shape[0] - 1)
		cv2.rectangle(frame, rec_start, rec_end, (0,0,255))
		out.write(frame)
	else:
		break

cap.release()
out.release()
cv2.destroyAllWindows()


exit()
