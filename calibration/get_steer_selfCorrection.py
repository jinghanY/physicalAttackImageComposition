import numpy as np
import pickle
import os
import math
import argparse
import random

def tf_rad2deg(rad):
	pi_on_180 = 0.017453292519943295
	return rad / pi_on_180

def deg2rad(deg):	
	return deg*np.pi/180.

parser = argparse.ArgumentParser()
parser.add_argument('--noiseInput','-n',help="trajectory noise")
parser.add_argument('--trajectoryNum','-j',help="trajectory num")
parser.add_argument('--task','-t',help="trajectory num")
parser.add_argument('--intersection','-i',help="trajectory num")
parser.add_argument('--min','-a')
parser.add_argument('--max','-b')
args =parser.parse_args()
noiseInput = int(args.noiseInput)
trajectoryNum = int(args.trajectoryNum)
min_frame = int(args.min)
max_frame = int(args.max)
task = args.task
intersection = args.intersection 
dataset_pt = "datasets_"+str(trajectoryNum)+"/"+task+"/"+intersection+"/"
control_info_pt = dataset_pt + "control_input/" 
frame_count_files = os.listdir(control_info_pt)
frame_count_files.sort(key=lambda fname:int(fname.split(".")[0]))
try:
	noises = np.random.randint(a,b,max_frame-min_frame+1)
except:
	print("Please define the range of your random noise first: a,b")
## 
# for example a = -10, b = 0, noises = np.random.randint(-10,0,max_frame-min_frame+1)
##

print(noises)
# get frame_counts
ct = 0
for frame_count_file in frame_count_files:
	frame_num = int(frame_count_file.split(".")[0])
	control_info_file = control_info_pt + frame_count_file 
	with open(control_info_file,"rb") as handle:
		control_dict = pickle.load(handle)
	steer = control_dict["steer"]
	if  min_frame <= frame_num <= max_frame: 
		steer = steer-deg2rad(noises[ct])
		ct = ct+1
	control_dict['steer'] = steer
	with open(control_info_file,"wb") as handle:
		pickle.dump(control_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)
		
