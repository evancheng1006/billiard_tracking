# https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from hu_moment import hu_moment
import os
import sys


from imshow_scaling import imshow_scaling
	
	
def get_hu_moment_features(img):
	height, width = img.shape[0], img.shape[1]
	if len(img.shape) == 2:
		img = img.reshape([height, width, 1])
	features = []
	for c in range(img.shape[2]):
		hus = hu_moment(img[:,:,c])
		features.extend(hus)
	return features


def euclidean_cluster_extraction(feature_vectors, min_dist, min_cluster_size):
	# bottle neck
	# m points n dim: feature_vectors.shape = (m,n)
	labels = [-1 for i in range(feature_vectors.shape[0])]
	# find connected components in a graph
	# create adjacent list
	adj_list = [[] for i in range(feature_vectors.shape[0])]
	for i in range(feature_vectors.shape[0]):
		if i % 30 == 0:
			print('ball_clustering, euclidean_cluster_extraction, progress: %.2f%%(%d/%d)' % (100*i/feature_vectors.shape[0], i, feature_vectors.shape[0]))
		for j in range(feature_vectors.shape[0]):
			if i != j:
				dist = np.linalg.norm(feature_vectors[i,:] - feature_vectors[j,:])
				if dist < min_dist:
					adj_list[i].append(j)
	# dfs find spanning forest
	tmp_label = 0
	min_unvisited = 0
	clusters = {}
	while min_unvisited < len(labels):
		stack = [min_unvisited]
		# non-recursive DFS
		while len(stack) > 0:
			s = stack[-1]
			stack.pop()
			if labels[s] == -1:
				labels[s] = tmp_label
			for t in adj_list[s]:
				if labels[t] == -1:
					stack.append(t)
		clusters[tmp_label] = [i for i in range(len(labels)) if labels[i] == tmp_label]
		while labels[min_unvisited] != -1:
			min_unvisited += 1
			if min_unvisited >= len(labels):
				break
		tmp_label += 1
	# filter cluster by min_cluster_size
	clusters = {label:clusters[label] for label in clusters if len(clusters[label]) >= min_cluster_size}
	# relabeling, sort by cluster size
	old_labels = list(clusters.keys())
	old_labels = sorted(old_labels, key = lambda old_label : len(clusters[old_label]), reverse=True)
	clusters = [sorted(clusters[old_labels[i]]) for i in range(len(old_labels))]
	return clusters


def ball_clustering(input_dir, output_dir, output_ext='bmp'):
	#output_ext recommended bmp or jpg
	if input_dir == output_dir:
		print('ball_clustering error: input_dir == output_dir')
		exit()
	
	all_fns = sorted([fn for fn in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, fn))])
	image_fns = [fn for fn in all_fns if fn.startswith('image_')]
	mask_fns = [fn for fn in all_fns if fn.startswith('mask_')]
	
	bh1 = []
	gh1 = []
	rh1 = []
	for i in range(len(image_fns)):
		fn = image_fns[i]
		mask = cv2.imread(os.path.join(input_dir,fn.replace('image','mask'))).astype('int')[:,:,0] / 255
		img = cv2.imread(os.path.join(input_dir,fn))
		
		features = get_hu_moment_features(img)
		bh1.append(1000*features[0])
		gh1.append(1000*features[7])
		rh1.append(1000*features[14])


	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(bh1, gh1, rh1)
	#plt.show()
	min_cluster_size = max(min(int(len(image_fns)/100), 15), 6)
	
	feature_vectors = np.stack([np.array(bh1), np.array(gh1), np.array(rh1)], axis=1)
	clusters = euclidean_cluster_extraction(feature_vectors, 0.05, min_cluster_size)
	#clusters is a list of lists
	for i in range(len(clusters)):
		cluster = clusters[i]
		for j in range(len(cluster)):
			fn_img_src = os.path.join(input_dir, image_fns[cluster[j]])
			fn_img_dst = os.path.join(output_dir, 'auto-template-image_%d_%d.%s' % (i, j, output_ext))
			fn_mask_src = os.path.join(input_dir, mask_fns[cluster[j]])
			fn_mask_dst = os.path.join(output_dir, 'auto-template-mask_%d_%d.%s' % (i, j, output_ext))
			cv2.imwrite(fn_img_dst, cv2.imread(fn_img_src))
			cv2.imwrite(fn_mask_dst, cv2.imread(fn_mask_src))
			#print(fn_img_dst + ' saved.')
			#print(fn_mask_dst + ' saved.')
			
	return
