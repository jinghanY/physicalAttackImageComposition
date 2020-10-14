import numpy as np
from shutil import copyfile
import shutil
import os 
import random
import glob
import matplotlib.pyplot as plt
import copy
import cv2
import pickle
from frame_homography import *
from tqdm import tqdm

# warping
def warp(im_src, im_dst, h, angle,frameNum):
	plt.imsave("/home/jinghan/Documents/auto_adv/carla-cluster/test/"+str(angle)+"/"+"source.png",im_src)
	im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
	im_res = copy.copy(im_dst)
	height, width = im_out.shape[0], im_out.shape[1]
	for j in range(height):
		for i in range(width):
			if np.max(im_out[j,i,:]) >= 0.5:
				im_res[j,i,:] = im_out[j,i,:]
	plt.imsave("/home/jinghan/Documents/auto_adv/carla-cluster/test/"+str(angle)+"/"+str(frameNum)+"_warp.png",im_res)


frames_root_path = "../../../../_out/"
points_path = os.listdir(frames_root_path)
points_path.sort(key=lambda fname: int(fname.split('t')[1]))
points_frames_path = []
for i in range(len(points_path)):
	point_path = points_path[i]
	points_frames_path.append(frames_root_path+point_path+"/")

# frames numbers that we want to save h matrix
framesNumbers = [90,91]
pointsAmount = 100

# get the initial points directory
file_initialPoints_pt = "/home/jinghan/Documents/auto_adv/carla-cluster/PythonClient-cluster/adversary_data/Town01_nemesis/single-line/adversaries/"

angle_range = [2,3]

src_image_pt = "/home/jinghan/Documents/auto_adv/carla-cluster/PythonClient-cluster/adversary_data/Town01_nemesis/single-line/adversaries/"
dst_image_pt = "/home/jinghan/Documents/auto_adv/carla-cluster/test/"

framesInfo_pt = "../../../../framesInfo/"


for i in tqdm(range(len(framesNumbers))):
	frameNum = framesNumbers[i]
	frameInfo = Frame(frameNum, pointsAmount,file_initialPoints_pt,points_frames_path,angle_range)
	frameInfo_file = framesInfo_pt + str(frameNum)+".pickle"
	with open(frameInfo_file,'wb') as handle:
		pickle.dump(frameInfo, handle, protocol=pickle.HIGHEST_PROTOCOL)
	#with open(frameInfo_file,'rb') as handle:
	#	frameInfo = pickle.load(handle)

	# test the correctness of the H matrix
	#for angle in angle_range:
	#	h_frame_angle = frame_this.hs[angle]
	#	im_src_pt = src_image_pt + "00"+str(angle-1)+".png"
	#	im_dst_pt = dst_image_pt + str(angle)+"/"+str(frameNum)+".png"
	#	im_src = cv2.cvtColor(cv2.imread(im_src_pt),cv2.COLOR_BGR2RGB)
	#	im_dst = cv2.cvtColor(cv2.imread(im_dst_pt),cv2.COLOR_BGR2RGB)
	#	im_src = np.multiply(im_src, 1.0/255.0)
	#	im_dst = np.multiply(im_dst, 1.0/255.0)
	#	warp(im_src, im_dst,h_frame_angle, angle, frameNum)



	


