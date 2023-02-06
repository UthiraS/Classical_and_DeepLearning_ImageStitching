#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2 as cv
import os
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
import math
# Add any python libraries here
import sys
import copy
from random import sample

"""
Plot Images

"""

def Plot_Images(img,coordinates):
	fig, axes = plt.subplots(1, 2, figsize=(8, 2), sharex=True, sharey=True)
	ax = axes.ravel()
	ax[0].imshow(img, cmap=plt.cm.gray)
	ax[0].axis('off')
	ax[0].set_title('Original')

	# ax[1].imshow(image_max, cmap=plt.cm.gray)
	# ax[1].axis('off')
	# ax[1].set_title('Maximum filter')

	ax[1].imshow(img, cmap=plt.cm.gray)
	ax[1].autoscale(False)
	ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
	ax[1].axis('off')
	ax[1].set_title('Peak local max')

	fig.tight_layout()

	plt.show()
	plt.close()
	
"""
Corner Detection
Save Corner detection output as corners.png
"""

def detectCorners(img):
	##Using Harris_Corners 
	# filename = 'chessboard.png'
	# img = cv2.imread(filename)
	gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
	gray = np.float32(gray)
	dst = cv.cornerHarris(gray,2,3,0.04)
	
	# print("Output of Corner Harris :",dst)
	#result is dilated for marking the corners, not important
	# dst = cv.dilate(dst,None)
	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()]=[255,0,0] #rgb
	# plt.imshow(img)
	# plt.show()
	# plt.close()
	return dst

def findPeakLocalMaxima(img):
	"""
	Finding Local Mamixa from the corners detected from harriscorner
	"""
	#method 1
	# im = img_as_float(img)
	# image_max = ndi.maximum_filter(dst, size=20, mode='constant')

	# Comparison between image_max and im to find the coordinates of local maxima
	# mean = np.mean(dst,dtype=np.float32)
	# std = np.sqrt((dst - mean)**2).mean()
	# dst = (dst - mean)/std
	#method 2
	coordinates = peak_local_max(img, min_distance=10)

	return coordinates

# display results


def ANMS(img,cimg,coordinates,n_best):

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
	# img[coordinates[:, 1], coordinates[:, 0]]
	# print("coordinates[:,1].shape :",coordinates[:,1].shape)
	# print("coordinates[:,0].shape :",coordinates[:,0].shape)
	# anms_img = []
	# anms_corners = []
	# N_best = 300
	# n_best = 300
	n_strong= len(coordinates)
	x=np.zeros((n_strong,1))
	y=np.zeros((n_strong,1))
	eDistance = 0
	# print("Nstrong :",n_strong)
	# inf = sys.maxsize
	r = np.ones(n_strong) * np.inf
	# r= np.array(r)
	# print(r)
	# print("",)
	# print("coordinates[:,0] :",coordinates[:,0])
	for i in range(n_strong):
		for j in range(n_strong):
			x_i = coordinates[i][0]
			y_i = coordinates[i][1]
			x_j = coordinates[j][0]
			y_j = coordinates[j][1]
			# print("x_i,y_i :",x_i,y_i)
			# print("x_j,y_j :",x_j,y_j)
			if(cimg[x_j,y_j]>cimg[x_i,y_i]):
				eDistance = np.square(x_i- x_j)+ np.square(y_i-y_j)
				# print(eDistance)
			if(r[i] > eDistance):
				r[i] =eDistance
				x[i] = x_j
				y[i] = y_j
	
	# sorting all the indices of r and fliping them to find the minimum 
	sort_distances_indices = np.flip(np.argsort(r))
	print(f"Taking {n_best} out of {sort_distances_indices.shape}")
	sort_distances_indices = sort_distances_indices[0:n_best]
	# print(sort_distances_indices)
	x_best=np.zeros((n_best,1))
	y_best=np.zeros((n_best,1))


	
	if x.shape[0] < n_best:
		n_best = x.shape[0]

	for i in range(n_best):
		x_best[i] = np.int0(y[sort_distances_indices[i]])
		y_best[i] = np.int0(x[sort_distances_indices[i]]) 
	anms_corner = np.int0(np.concatenate((x_best, y_best), axis = 1))
	print("Anms corner length :",len(anms_corner))
	print("Anms corner :",anms_corner)
	# x_coord = y[0:sort_distances_indices]
	# y_coord = x[0:sort_distances_indices]
	# anms_corner = np.int0(np.concatenate((x_coord, y_coord ), axis = 1))
	# best_points = coordinates[anms_corner]

	# fig, axes = plt.subplots(1, 2, figsize=(8, 2), sharex=True, sharey=True)
	# ax = axes.ravel()
	# ax[0].imshow(img, cmap=plt.cm.gray)
	# ax[0].axis('off')
	# ax[0].set_title('Original')

	

	# ax[1].imshow(img, cmap=plt.cm.gray)
	# ax[1].autoscale(False)
	# ax[1].plot(anms_corner[:,0], anms_corner[:,1], 'r.')
	# ax[1].axis('off')
	# ax[1].set_title('ANMS')

	# fig.tight_layout()

	# plt.show()
	# plt.close()
	
	return anms_corner


"""
Feature Descriptors
Save Feature Descriptor output as FD.png
"""

def featureDes(img,coordinates,p_size):
	img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	corner_patches =[]
	n_descriptors =[]
	patches =[]
	p_size =40
	for y,x in coordinates:
		# print(x,y)
		gray = copy.deepcopy(img_gray)
		gray = np.pad(img_gray, ((p_size,p_size), (p_size,p_size)), mode='constant', constant_values=0)
		# print("Gray Shape :",gray.shape)
		x_start = int(x + p_size/2)
		y_start = int(y + p_size/2)
		if((x_start >(gray.shape[0]-40)) or (y_start >(gray.shape[1]-40))):
			continue
		# print("START :",x_start,y_start)
		descriptor = gray[x_start:x_start+p_size, y_start:y_start+p_size]  
		# print(descriptor.shape)
		descriptor = cv.GaussianBlur(descriptor, (7,7), cv.BORDER_DEFAULT)   
		# print(descriptor.shape)               #apply gaussian blur
		descriptor = descriptor[::5,::5]														#sub sampling to 8x8
		# print(descriptor.shape)
		descriptor_1 = descriptor.reshape((64,1))
		descriptor_standard = (descriptor_1 - descriptor_1.mean())/descriptor_1.std()
		n_descriptors.append(descriptor_standard)
		# patch = img_gray[y-20:y+20, x-20:x+20]
		# patch = cv.blur(patch,(5,5), 0)
		# # print(patch.shape)
		# patch = cv.resize(patch, (8,8), interpolation = cv.INTER_AREA)
		# # print(patch.shape)
		# # patch = patch[::5,::5]		
		# patches.append(patch)
		# print(patch.shape)												#sub sampling to 8x8
		# corner_patch = patch.reshape((64,1))
		# print(corner_patch.shape)
		# corner_patch= (corner_patch - corner_patch.mean())/corner_patch.std()
		# print("corner_patch :",corner_patch)
		# corner_patches.append(corner_patch)
			# patches.append(patch)
		# cv.imshow('patch',patch)
		# cv.waitKey(0)
		# cv.close()
	# print("patches",patches)
	# patches_x =[]
	# patches_y =[]
	# for i in range(len(patches)):
	# 	patches_x.append(patches[i][:,0])
	# 	patches_y.append(patches[i][:,1])
	


	# fig, axes = plt.subplots(1, 1, figsize=(8, 2), sharex=True, sharey=True)
	# # ax = axes.ravel()
	# # ax[0].imshow(img, cmap=plt.cm.gray)
	# # ax[0].axis('off')
	# # ax[0].set_title('Original')

	

	# axes.imshow(img, cmap=plt.cm.gray)
	# axes.autoscale(False)
	# axes.plot(patches_x, patches_y, 'r.')
	# axes.axis('off')
	# axes.set_title('Patches')

	return n_descriptors

# # fig.tight_layout()

# plt.show()
# plt.close()

"""
	Feature Matching
	Save Feature Matching output as matching.png
"""

def featureMatching(images,feature_vectors,best_corners,match_threshold):

	print("Feature Matching and Plotting :")
	keypoints1 =[]
	keypoints2 =[]
	distances =[]
	corners1= best_corners[0]
	corners2 = best_corners[1]
	feature_vector_1 = feature_vectors[0]
	feature_vector_2 = feature_vectors[1]
	i = 0
	matched_pairs = []
	for fpoint1  in  feature_vector_1:
		min_dist = math.inf
		second_min_dist = math.inf
		key_point_2 = None
		diffs =[]
		for fpoint2  in feature_vector_2:
			distance = np.sum((fpoint1 - fpoint2)**2)
			# sqr_diff.append(diff)
			diffs.append(distance)
		diffs = np.array(diffs)
		diff_sort = np.argsort(diffs)
		sqr_diff_sorted = diffs[diff_sort]
		ratio = sqr_diff_sorted[0]/(sqr_diff_sorted[1])
		# ratio = min_dist/second_min_dist
		# top_matche = np.argmin(diffs)
		if(ratio<match_threshold):
			
			distances.append(min_dist)
			matched_pairs.append((corners1[i,0:3], corners2[diff_sort[0],0:3]))
	
		i = i+1
	print("Length of matched pairs :",len(matched_pairs))
	# keypoints1 = np.array(keypoints1)
	# keypoints2 = np.array(keypoints2)
	keypoints1 = [x[0] for x in matched_pairs]
	keypoints2 = [x[1] for x in matched_pairs]
	kpoints1=[]
	for i in range(len(keypoints1)):
		kpoints1.append(cv.KeyPoint(int(keypoints1[i][0]), int(keypoints1[i][1]), 3))
	kpoints2=[]
	for i in range(len(keypoints2)):
		kpoints2.append(cv.KeyPoint(int(keypoints2[i][0]), int(keypoints2[i][1]), 3))

	matched_pairs_idx =  [(i,i) for i,j in enumerate(matched_pairs)]
	m = []
	for i in range(len(matched_pairs_idx)):
		m.append(cv.DMatch(int(matched_pairs_idx[i][0]), int(matched_pairs_idx[i][1]), 2))
	matches = m
	# i= 0
	# for distance in distances:
	# 	matches.append(cv.DMatch(i,i,2))
	# 	i = i+1

	ret = np.array([])
	out = cv.drawMatches(img1=images[0],
        keypoints1=kpoints1,
        img2=images[1],
        keypoints2=kpoints2,
        matches1to2=matches,outImg = None,flags =2)
	# plt.imshow(out)
	# plt.show()
	# plt.close()
	
	
	
	return matched_pairs,distances

"""
Refine: RANSAC, Estimate Homography
"""
def rANSAC(images,matched_pairs, threshold, NMAX):

	index =[]
	inliers =[]
	for _ in range(NMAX):

		# Matched Points from Feature Mapping
		matchedPoints1 = [x[0] for x in matched_pairs]
		matchedPoints2 = [x[1] for x in matched_pairs]
		
		#Step 1; Choosing four random points out of all matched points
		length= range(len(matched_pairs))
		random4PointIndex = sample(length,4)
		matched_pairs= np.asarray(matched_pairs)
		
		randomPoints1 = [matchedPoints1[i] for i in random4PointIndex]
		randomPoints2 = [matchedPoints2[i] for i in random4PointIndex]
		#Step 2; Finding the homogrpahy betwene the two points
		H = cv.getPerspectiveTransform(np.float32(randomPoints1),np.float32(randomPoints2))
		
		points_iterations =[]
		count_iteration = 0
		#Step 3: obtaining SSd between PI_dash and dot product between H and PI
		for i in length:
				# Reshaping both the keypoints to (2,1)
				PI_dash = np.expand_dims(np.array(matchedPoints2[i]), 1)			
				PI = np.expand_dims(np.array(matchedPoints1[i]), 1)
				# The shape of H matrix is 3*3 , hence changing the shape of PI to 3,1 by stacking ones in thirs vertical axis
				PI =  np.vstack([PI, 1])
				
				# Finding HPI by dot profuct betwen H (3*3) and PI (3*1), rsulting 3*1
				HPI = np.dot(H,PI)
				# Normalising HPI 
				# print(HPI.shape)
				# if HPI[2]!=0:
				# 	HPI = HPI/HPI[2]
				# else:
				# 	HPI = HPI/0.000001
				# #Reshaping the dimenion of HPI to be of same shape of PI_dash
				HPI = HPI[0:2,:]
				#Computing SSD 
				SSD = np.linalg.norm(PI_dash-HPI)
				# print("SSD :",SSD)
				if(SSD<threshold):
					count_iteration += 1
					points_iterations.append((matchedPoints1[i],matchedPoints2[i]))
		index.append(count_iteration)	
		inliers.append(points_iterations)
	count_length= len(index)
	index = np.argsort(index)
	max_index = index[count_length-1]
	# Selecting the points which least ssd
	# print("inliers :",inliers)
	final_matches = inliers[max_index]
	# print("final_matches :",final_matches)
	final_pts_1 = [x[0] for x in final_matches]
	final_pts_2 = [x[1] for x in final_matches]
	H_matrix, status = cv.findHomography(np.float32(final_pts_1),np.float32(final_pts_2))


	# print("Length of matched pairs :",len(final_matches))
	# keypoints1 = np.array(keypoints1)
	# keypoints2 = np.array(keypoints2)
	# keypoints1 = [x[0] for x in matched_pairs]
	# keypoints2 = [x[1] for x in matched_pairs]
	fpoints1=[]
	for i in range(len(final_pts_1)):
		fpoints1.append(cv.KeyPoint(int(final_pts_1[i][0]), int(final_pts_1[i][1]), 3))
	fpoints2=[]
	for i in range(len(final_pts_2)):
		fpoints2.append(cv.KeyPoint(int(final_pts_2[i][0]), int(final_pts_2[i][1]), 3))

	final_pairs_idx =  [(i,i) for i,j in enumerate(final_matches)]
	m = []
	for i in range(len(final_pairs_idx)):
		m.append(cv.DMatch(int(final_pairs_idx[i][0]), int(final_pairs_idx[i][1]), 2))
	matches = m
	# i= 0
	# for distance in distances:
	# 	matches.append(cv.DMatch(i,i,2))
	# 	i = i+1

	ret = np.array([])
	out = cv.drawMatches(img1=images[1],
        keypoints1=fpoints2,
        img2=images[0],
        keypoints2=fpoints1,
        matches1to2=matches,outImg = None,flags =2)
	plt.imshow(out)
	plt.show()
	plt.close()
	return H_matrix,final_matches


def findHomography(imageA,imageB):
		
	images =[]	
	best_corners =[]
	feature_vectors =[]
	# image_files = os.listdir(folder_name)
	NBEST = 2000
	patch_Size = 40
	#ratio threshold for feature mapping
	match_threshold = 0.2
	#distance threshold and number of iterations for RANSAC
	threshold = 10
	NMAX = 5000
	
	images.append(imageA)
	images.append(imageB)
	for img in images:
		corner_img = detectCorners(img)
		coords = findPeakLocalMaxima(corner_img)
		anms_corner = ANMS(copy.deepcopy(img),corner_img,coords,NBEST)
		feature_vector = featureDes(copy.deepcopy(img),anms_corner,patch_Size) 
		best_corners.append(anms_corner)
		feature_vectors.append(feature_vector)
	matched_pairs,distances = featureMatching(images,feature_vectors,best_corners,match_threshold)	
	HMatrix, final_matches = rANSAC(images,matched_pairs, threshold, NMAX)

	return HMatrix




def StitchImages(images, H):

	img1 = copy.deepcopy(images[1])
	img2 = copy.deepcopy(images[0])
	# Getting the shape of two images
	h1,w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]
	# Getting the corner points  of two images
	corner_pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
	corner_pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
	# warping the corner points of image 2 based on the known homography
	corner_pts2_warped = cv.perspectiveTransform(corner_pts2, H)
	# finding all the points from corners of image 1 and corners of warped image2 
	pts = np.concatenate((corner_pts1, corner_pts2_warped), axis=0)

	# finding the minimum 4 points out of all the corner points
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel())
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel())
	
	# fixing the new origin for the stitched image
	Ht = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]]) 
	# warping the image 2 based on homography translated to new origin from xmin-xmax and ymin-ymax 
	result = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin), flags = cv.INTER_LINEAR)
	# from the obtained warped image, fill the cooridnates from -ymin to h1-ymin and -xmin to w1-xmin as image1 in order to combine
	result[(-ymin):h1+(-ymin),(-xmin):w1+(-xmin)] = img1

			
	# plt.imshow(result)
	# plt.show()
	# plt.close()
	
	return result




def stitching():
	folder_name = "/home/uthira/usivaraman_p1/Phase1/Code/Data/Set1/"
	imagesset =[]
	gray_images =[]
	best_corners =[]
	feature_vectors =[]
	image_files = os.listdir(folder_name)
	NBEST = 1500
	patch_Size = 40
	gray_images =[]
	
	#ratio threshold for feature mapping
	match_threshold = 0.2
	#distance threshold and number of iterations for RANSAC
	threshold = 15
	NMAX = 8000
	#Finding and storing feature vectors, best corners for all images
	i = 0
	for img_name in image_files:
		print(i)
		print("image name",img_name)
		img_path = folder_name + img_name
		img = cv.imread(img_path)
		gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		imagesset.append(img)
		gray_images.append(gray_img)
		i = i+ 1
		# corner_img = detectCorners(img)
		# coords = findPeakLocalMaxima(corner_img)
		# anms_corner = ANMS(copy.deepcopy(img),corner_img,coords,NBEST)
		# feature_vector = featureDes(copy.deepcopy(img),anms_corner,patch_Size) 
		# best_corners.append(anms_corner)
		# feature_vectors.append(feature_vector)
	

	num_images = len(image_files)
	num_pairs = num_images -1
	print("num_pairs : ",num_pairs)
	
	
	index1 = 0
	index2 = 0
	
	for pair_index in range(num_pairs):
		
		best_corners =[]
		feature_vectors =[]
		index2 = pair_index+1
		images = [imagesset[index1], imagesset[index2]]
		
		
		for i in range(2):
			img = images[i]
			
			corner_img = detectCorners(img)
			coords = findPeakLocalMaxima(corner_img)
			anms_corner = ANMS(copy.deepcopy(img),corner_img,coords,NBEST)
			feature_vector = featureDes(copy.deepcopy(img),anms_corner,patch_Size) 
			best_corners.append(anms_corner)
			feature_vectors.append(feature_vector)
	
		
		matched_pairs,distances = featureMatching(images,feature_vectors,best_corners,match_threshold)		
		
		index1 = index2
			
		HMatrix, final_matches = rANSAC(images,matched_pairs, threshold, NMAX)
			
		warped = StitchImages(images, HMatrix)	

		imagesset[index1] = warped

		result = warped
	
	print("Final Output :")
	plt.imshow(result)
	plt.savefig("/home/uthira/usivaraman_p1/Phase1/Code/Result/Set1/panaroma.jpg")
	plt.show()
	plt.close()
		

	

	
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

def main():

	stitching()
	
if __name__ == "__main__":
    main()
