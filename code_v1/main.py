import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import shutil # zip file


from mkdir_p import mkdir_p
from auto_background import auto_background
from ball_extraction import ball_extraction
from ball_clustering import ball_clustering
from tracking_from_auto_template import tracking_from_auto_template
from create_tracking_video_from_detection_result_csv import create_tracking_video_from_detection_result_csv


def make_clean_dir(path):
	mkdir_p(path)
	shutil.rmtree(path) # dangerous
	mkdir_p(path)
	return


def main():
	input_fn = sys.argv[1]
	#input_fn = 'clip009.mp4'
	#input_fn = 'clip010.mp4'
	
	output_fn = os.path.splitext(input_fn)[0] + '_output.mp4'
	output_tracking_result_fn = os.path.splitext(input_fn)[0] + '_tracking_result.csv'
	output_background_fn = os.path.splitext(input_fn)[0] + '_background.bmp'
	tmp_dir = '__tmp/'
	tmp_dir_cluster = os.path.join(tmp_dir, 'cluster/')
	tmp_dir_extraction = os.path.join(tmp_dir, 'extraction/')
	output_temp_file_zip = input_fn + '.__tmp'
	mkdir_p(tmp_dir)
	make_clean_dir(tmp_dir_cluster) # dangerous
	make_clean_dir(tmp_dir_extraction) # dangerous
	print('auto_background starts...')
	#dangerous: race condition(ignore it)
	if os.path.basename(output_background_fn) in os.listdir('.'):
		print('use cached background file: ' + output_background_fn)
	else:
		auto_background(input_fn, output_background_fn)
	print('auto_background finished')
	print('ball_extraction starts...')
	ball_extraction(input_fn, output_background_fn, tmp_dir_extraction)
	print('ball_extraction finished')
	print('ball_clustering starts...')
	ball_clustering(tmp_dir_extraction, tmp_dir_cluster, 'bmp')
	print('ball_clustering finished')
	print('tracking_from_auto_template starts...')
	if os.path.basename(output_tracking_result_fn) in os.listdir('.'):
		print('use cached tracking result csv file: ' + output_tracking_result_fn)
	else:
		tracking_from_auto_template(input_fn, output_background_fn, tmp_dir_cluster, output_tracking_result_fn)
	print('tracking_from_auto_template finished')
	print('creating output video...')
	create_tracking_video_from_detection_result_csv(input_fn, output_tracking_result_fn, output_fn)
	print('output video %s created' % output_fn)
	print('creating temp file archive...')
	shutil.make_archive(output_temp_file_zip, 'zip', tmp_dir)
	print('temp file archive %s.zip created' % output_temp_file_zip)
	return


main()