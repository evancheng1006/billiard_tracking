import cv2
import csv
import numpy as np
import time
import matplotlib.pyplot as plt


def get_frames(fn):
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


def read_detection_result_csv(fn):
	import re
	legal_char_regex = r'^[\d,:\(\)\{\}]*$'
	result = {}
	with open(fn, newline='') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			iFrame = int(row['frame_num'])
			tmp = row['detection_result'].strip()
			matched = re.findall(legal_char_regex, tmp)
			if len(matched) == 0:
				print('Error: wrong csv format %s' % tmp)
				exit()
			result[iFrame] = eval(matched[0]) # dangerous
	return result

	
def create_tracking_video_from_detection_result_csv(input_fn, detection_result_csv_fn, output_fn):
	if input_fn == detection_result_csv_fn:
		raise ValueError('Error: input_fn == detection_result_csv_fn')
	if output_fn == input_fn:
		raise ValueError('Error: output_fn == input_fn')
	if output_fn == detection_result_csv_fn:
		raise ValueError('Error: output_fn == detection_result_csv_fn')
		
	result = read_detection_result_csv(detection_result_csv_fn)
	
	frames, width, height, fps = get_frames(input_fn)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_fn,fourcc, fps, (width,height))
	
	for i in range(len(frames)):
		tmp = frames[i].copy()
		if i % 500 == 0:
			print('video generation progress: %.2f%%(%d/%d)' % (100*i/len(frames), i, len(frames)))
		if i in result.keys():
			for label in result[i]:
				ind, matched_size = result[i][label]
				rec_start = (ind[1], ind[0])
				rec_end = (ind[1]+matched_size[1]-1, ind[0]+matched_size[0]-1)
				text_loc = (max(ind[1]-6,0), max(ind[0]-6,0))
				cv2.rectangle(tmp, rec_start, rec_end, (0,0,255))
				cv2.putText(tmp, str(label), text_loc, cv2.FONT_HERSHEY_SIMPLEX,
					0.4, (0,255,255), 1, cv2.LINE_AA)
		out.write(tmp)
	
	out.release()
	return