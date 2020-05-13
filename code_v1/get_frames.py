def get_frames(fn):
	import cv2
	import numpy as np
	cap = cv2.VideoCapture(fn)
	if cap.isOpened():
		width  = int(cap.get(3))
		height = int(cap.get(4))
		fps = cap.get(5) # float
		frame_count = int(cap.get(7))
	else:
		raise ValueError('cannot open video file %s.' % fn)
	frames = {}
	i_frame = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			frames[i_frame] = frame.copy()
			i_frame += 1
		else:
			break
	cap.release()
	frames = [frames[i] for i in range(i_frame)]
	return frames, width, height, fps