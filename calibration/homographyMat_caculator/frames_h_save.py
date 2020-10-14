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
import argparse

def mkdir_my(pathName):
	if not os.path.exists(pathName):
		os.mkdir(pathName)

def copy_clean_frames(src,dst):
	frames_to_copy =glob.glob(src+"*.png")
	frames_to_copy.sort(key=lambda fname: int(fname.split('/')[-1].split('.')[0]))
	frames_to_copy = [x.split('/')[-1] for x in frames_to_copy]
	for i in range(len(frames_to_copy)):
		src_file = src + frames_to_copy[i]
		dst_file = dst + frames_to_copy[i]
		shutil.copyfile(src_file, dst_file)

# Task: get the valid_points

parser = argparse.ArgumentParser()
parser.add_argument('--task','-t',help="which task")
parser.add_argument('--intersection','-i',help="which intersection")
parser.add_argument('--frame_min','-l',help="min frame number")
parser.add_argument('--frame_max','-z',help="max frame number")
parser.add_argument('--trajectory','-j',help='trajectory')

args =parser.parse_args()
frames_min = int(args.frame_min)
frames_max = int(args.frame_max)
task = args.task+"/"
intersection =args.intersection+ "/"
trajectory_no = int(args.trajectory)

#frames_min = 75
#frames_max = 76
#task = "intersection-left"+"/"
#intersection = "100_96"+"/"

frames_root_path_radius1 = "out_"+intersection.strip("/")+"_"+str(trajectory_no)+"/"
points_path = glob.glob(frames_root_path_radius1+"point*")
points_path.sort(key=lambda fname: int(fname.split('int')[1]))
valid_points = [int(x.split('int')[1]) for x in points_path]

valid_points = np.array(valid_points)
valid_points.sort()
points_frames_path_radius1 = []
for i in range(len(valid_points)):
	points_frames_path_radius1.append(frames_root_path_radius1 + "point"+str(valid_points[i])+"/")

dataset_pt = "datasets_"+str(trajectory_no)+"/"+task+intersection 

adversary_angles_pt = dataset_pt + "adversary/"
framesInfo_pt = dataset_pt + "framesInfo/"
clean_frames_pt = dataset_pt + "clean_frame/"

#mkdir_my(adversary_angles_pt)
#mkdir_my(framesInfo_pt)
#mkdir_my(clean_frames_pt)

frames_range = list(range(frames_min, frames_max))
file_initialPoints_pt = "adversary/"

#copy_clean_frames("out/"+"baseline/",clean_frames_pt)
for frameNum in tqdm(frames_range):
	print("begin to get the frameInfo...")
	frameInfo = Frame(frameNum,valid_points,file_initialPoints_pt,points_frames_path_radius1)
	frameInfo_rightAngle = {}
	print(frameInfo.hs_cv2)
	#print(frameInfo.hs)
	frameInfo_rightAngle['h_cv2'] = frameInfo.hs_cv2['h'] 
	#frameInfo_rightAngle['h'] = frameInfo.hs['h'] 
		
	frameInfo_file = framesInfo_pt + str(frameNum)+".pickle" 
	
	print(frameInfo_rightAngle)
	with open(frameInfo_file,'wb') as handle:
		pickle.dump(frameInfo_rightAngle, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("finish dumping.")
