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
import os
import time
from skimage.io import imread, imsave
from copy import deepcopy
import shutil
import argparse
from adversary.shape_fly_simulator import Shape
import scipy
import scipy.misc
from utils import *

def mkdir_my(pathName):
	if not os.path.exists(pathName):
		os.mkdir(pathName)

# Task: get the valid_points

parser = argparse.ArgumentParser()
parser.add_argument('--task','-t',help="which task")
parser.add_argument('--intersection','-i',help="which intersection")
parser.add_argument('--frame_min','-l',help="min frame number")
parser.add_argument('--frame_max','-z',help="max frame number")
parser.add_argument('--trajectory','-j',help="trajectory number")

args =parser.parse_args()
frames_min = int(args.frame_min)
frames_max = int(args.frame_max)
trajectory_no = int(args.trajectory)
task = args.task+"/"
intersection =args.intersection+ "/"

dataset_pt = "datasets_"+str(trajectory_no)+"/"+task+intersection
adversary_angles_pt = dataset_pt + "adversary/"
framesInfo_pt = dataset_pt + "framesInfo/"
clean_frames_pt = dataset_pt + "clean_frame/"
save_pt = dataset_pt + "quality_check/"

frames_range = list(range(frames_min, frames_max))
clean_frames = []
framesInfos = []
frames_range_new = []
print(frames_range)
for i in range(len(frames_range)):
	frame_num= frames_range[i]
	frameInfo_file = framesInfo_pt + str(frame_num) + ".pickle"
	print(frameInfo_file)
	with open(frameInfo_file,"rb") as handle:
		frameInfo_this = pickle.load(handle)
		h = np.array(frameInfo_this["h_cv2"])

		print(h)

		if h.any() == None:
			continue
		elif np.isnan(h).any():
			continue
		framesInfos.append(h)
	frames_range_new.append(frame_num)
	clean_frame_file = clean_frames_pt + str(frame_num) + ".png"
	clean_frame = np.float32(imread(clean_frame_file))/255.
	clean_frames.append(clean_frame)
clean_frames = np.array(clean_frames)
framesInfos = np.array(framesInfos)

draw = Shape("Town01_nemesisA",'single-line',400,400,dataset_pt)
#numbers = {'rectangle1': {'rot': 0, 'pos': -10, 'width': 395, 'length': -5, 'r': 0, 'g': 0, 'b': 0}}
numbers = {'rectangle1': {'rot': 82.97999918460846, 'pos': 355.0000047683716, 'width': 4.760000109672546, 'length': 34.85999971628189, 'r': 76.78999751806259, 'g': 149.94999766349792, 'b': 92.52999722957611}, 'rectangle2': {'rot': 1.9400000385940075, 'pos': 228.89999747276306, 'width': 21.289999783039093, 'length': 3.500000052154064, 'r': 67.98999756574631, 'g': 27.240000665187836, 'b': 76.66999846696854}, 'rectangle3': {'rot': 16.06999933719635, 'pos': 336.55999302864075, 'width': 48.39000105857849, 'length': 36.679999232292175, 'r': 22.630000486969948, 'g': 132.25999474525452, 'b': 80.38000017404556}, 'rectangle4': {'rot': 51.96999907493591, 'pos': 4.770000167191029, 'width': 32.739999890327454, 'length': 2.669999934732914, 'r': 0.0, 'g': 0.0, 'b': 2.03000009059906},'rectangle5': {'rot': 79.56000030040741, 'pos': 99.88000512123108, 'width': 53.24000120162964, 'length': 27.33000010251999, 'r': 129.04000282287598, 'g': 33.890001475811005, 'b': 121.71000242233276}}

def get_adversary_frame(i):
	frame_num = frames_range_new[i]
	
	adversary_image = draw.draw_lines(numbers)
	adversary_image = np.float32(adversary_image)/255.
	
	h = copy.deepcopy(framesInfos[i])
	clean_frame = copy.deepcopy(clean_frames[i])

	h_my_info = prepWarp_crop_resize(adversary_image,h,(clean_frame.shape[1],clean_frame.shape[0]))
	im_out = doWarp(adversary_image,h_my_info,(clean_frame.shape[1],clean_frame.shape[0]))
	im_res = clean_frame
	indices = im_out[:,:,3]>0
	im_res[indices] = im_out[indices,:3]
	return im_res

for i in range(len(frames_range_new)):
	frame_num = frames_range_new[i]
	print(frame_num)
	image_warped_this = get_adversary_frame(i)
	image_warped_this = image_warped_this.astype(np.float32)
	scipy.misc.imsave(save_pt + str(frame_num)+".png",image_warped_this)
	#scipy.misc.imsave(str(frame_num)+".png",image_warped_this)
