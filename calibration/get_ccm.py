from color_image import *
import os
import numpy as np
import glob
import shutil
import copy
from shutil import copyfile
from skimage.io import imread,imsave
from shutil import copyfile
import argparse
from adversary.shape_test import Shape

def crop_black_region(adversary_image,h,frame_num,img,save_pt_crop,save_pt_black_region,color_num):
	h_my_info = prepWarp(adversary_image,h,(img.shape[1],img.shape[0]))
	oidx = h_my_info[0]
	inidx = h_my_info[1]
	canvas_image_croped = img[oidx[1],oidx[0],:]
	# black region
	res = copy.deepcopy(img)
	res[oidx[1],oidx[0],:] = 0
	imsave(save_pt_black_region+str(color_num)+".png",res)

def read_pixels(inFile):
	lines = open(inFile,"r")
	pixels = []
	for line in lines:
		line = line.strip("\n").split(",")
		pixel_this = list(map(int,line))
		pixels.append(pixel_this)
	return np.array(pixels)

parser = argparse.ArgumentParser()
parser.add_argument("--task","-t",type=str)
parser.add_argument("--intersection","-i",type=str)
parser.add_argument("--frameNumInput","-f",type=int)
args = parser.parse_args()

task=args.task
intersection=args.intersection
frame_num_input=args.frameNumInput 

dataset_pt = "datasets_1/"+task+"/"+intersection+"/"
try:
	os.mkdir(dataset_pt+"CCM/")
except:
	pass
try:
	os.mkdir(dataset_pt+"colors_in/")
except:
	pass
try:
	os.mkdir(dataset_pt+"colors_out/")
except:
	pass

color_pt = "color_"+intersection+"/"
color_nums_files = glob.glob(color_pt+"point*")
color_nums_files.sort(key=lambda fname:int(fname.split('point')[1]))
color_nums_files = color_nums_files[1:]
color_nums = [int(x.split("int")[1]) for x in color_nums_files]
color_nums = np.array(color_nums)
color_nums = np.delete(color_nums, [23,69])
frame_files_pt = dataset_pt + "framesInfo/"
frame_files = os.listdir(frame_files_pt)
frame_files.sort(key=lambda fname:int(fname.split(".")[0]))
frame_range = [int(x.split(".")[0]) for x in frame_files]
frame_range = [frame_num_input]
adversary_image = np.float32(imread("adversary/adversary_Town01_nemesisA.png"))/255.

color_pixel_in_pt = "adversary/color_map.txt"
pixel_in_all = read_pixels(color_pixel_in_pt)

framesInfo_pt = dataset_pt + "framesInfo/"
for frame_num in frame_range:
	colors_in = []
	colors_out = []
	print(frame_num)
	for color_num in color_nums:
		r_in,g_in,b_in = pixel_in_all[color_num-1]
		color_in = [r_in,g_in,b_in]
		colors_in.append(color_in)
		image_pt = "color_"+intersection+"/point"+str(color_num)+"/"
		dir_pt = "pixels_valid/point"+str(color_num) + "/"
		image_file = image_pt+str(frame_num)+".png"
		img = imread(image_file)
		img = np.float32(imread(image_pt+str(frame_num)+".png"))/255.

		# get h
		frameInfo_file = framesInfo_pt + str(frame_num) + ".pickle"
		with open(frameInfo_file,"rb") as handle:
			frameInfo_this = pickle.load(handle)
			h_my = np.array(frameInfo_this["h_cv2"])
	
		try:
			r_out,g_out,b_out = get_out_pixel_valid(adversary_image,h_my,frame_num,img)
			color_out = [r_out,g_out,b_out]
			colors_out.append(color_out)
		except:
			pass
	
	colors_in = np.array(colors_in)
	colors_out = np.array(colors_out)
	colors_in = np.float32(colors_in)/255.
	colors_in = np.vstack([(colors_in).T, np.ones(colors_in.shape[0])]).T
	try:	
		CCM = np.linalg.lstsq(colors_in,colors_out, rcond=None)[0] 
		np.savez(dataset_pt+"CCM/"+"ccm.npz", CCM=CCM)
	except:
		pass

# check for drawing
draw_save_pt = "check/"+intersection+"/"
draw_save_pt_bias = draw_save_pt + "bias/"
draw_save_pt_real = draw_save_pt + "real/"
draw_save_pt_crop = draw_save_pt + "crop/"
draw_save_pt_black_region_lastFrame = draw_save_pt + "black_region_lastFrame/"
draw_save_pt_black_region_firstFrame = draw_save_pt + "black_region_firstFrame/"

draw_save_pt1 = "check_lastFrame/"+intersection+"/"
draw_save_pt_bias1 = draw_save_pt1 + "bias/"
draw_save_pt_real1 = draw_save_pt1 + "real/"
draw_save_pt_crop1 = draw_save_pt1 + "crop/"
draw_save_pt_black_region_lastFrame1 = draw_save_pt1 + "black_region_lastFrame/"
draw_save_pt_black_region_firstFrame1 = draw_save_pt1 + "black_region_firstFrame/"

if os.path.exists(draw_save_pt):
	shutil.rmtree(draw_save_pt)

if not os.path.exists(draw_save_pt_bias):
	os.makedirs(draw_save_pt_bias)
if not os.path.exists(draw_save_pt_real):
	os.makedirs(draw_save_pt_real)
if not os.path.exists(draw_save_pt_crop):
	os.makedirs(draw_save_pt_crop)
if not os.path.exists(draw_save_pt_black_region_lastFrame):
	os.makedirs(draw_save_pt_black_region_lastFrame)
if not os.path.exists(draw_save_pt_black_region_firstFrame):
	os.makedirs(draw_save_pt_black_region_firstFrame)

if os.path.exists(draw_save_pt1):
	shutil.rmtree(draw_save_pt1)

if not os.path.exists(draw_save_pt_bias1):
	os.makedirs(draw_save_pt_bias1)
if not os.path.exists(draw_save_pt_real1):
	os.makedirs(draw_save_pt_real1)
if not os.path.exists(draw_save_pt_crop1):
	os.makedirs(draw_save_pt_crop1)
if not os.path.exists(draw_save_pt_black_region_lastFrame1):
	os.makedirs(draw_save_pt_black_region_lastFrame1)
if not os.path.exists(draw_save_pt_black_region_firstFrame1):
	os.makedirs(draw_save_pt_black_region_firstFrame1)

frame_num = frame_range[0]
clean_frame_file = "color_"+intersection+"/"+"point0/"+str(frame_num)+".png"
clean_frame = np.float32(imread(clean_frame_file))/255.
frameInfo_file = framesInfo_pt + str(frame_num) + ".pickle"

with open(frameInfo_file,"rb") as handle:
	frameInfo_this = pickle.load(handle)
	h_my = np.array(frameInfo_this["h_cv2"])

check_num_random = color_nums
CCM_pt = dataset_pt+"CCM/"+"ccm.npz"
CCM_npzfile = np.load(CCM_pt)
CCM = CCM_npzfile["CCM"]
colors_in = np.float32(pixel_in_all)/255.
colors_in = np.vstack([(colors_in).T, np.ones(colors_in.shape[0])]).T

colors_transformed = np.matmul(colors_in,CCM)
adversary = Shape(city_name="Town01_nemesisA")

color_this = colors_transformed[0]
color_this = np.int32(color_this*255.)
adversary.lines_rotate(color_this)
adversary_image = np.float32(imread("adversary/adversary_Town01_nemesisA.png"))/255.
	
# get crop canvas and black region
image_pt = "color_"+intersection+"/point"+str(color_num)+"/"
img = np.float32(imread(image_pt+str(frame_num)+".png"))/255.
crop_black_region(adversary_image,h_my,frame_num,img,draw_save_pt_crop,draw_save_pt_black_region_firstFrame,color_num)	


from utils_warping import *

for color_num in check_num_random:
	color_this = colors_transformed[color_num-1]
	color_this = np.int32(color_this*255.)
	adversary.lines_rotate(color_this)
	adversary_image = np.float32(imread("adversary/adversary_Town01_nemesisA.png"))/255.
	
	# bias
	h_my_info = prepWarp(adversary_image,h_my,(clean_frame.shape[1],clean_frame.shape[0]))
	im_out = doWarp(adversary_image,h_my_info,(clean_frame.shape[1],clean_frame.shape[0]))
	indices = im_out[:,:,3]>0
	im_res = copy.deepcopy(clean_frame)
	im_res[indices,:] = im_out[indices,:3]
	imsave(draw_save_pt_bias+str(color_num)+".png",im_res)

	# real
	src = "color_"+intersection+"/" +"point"+str(color_num)+"/" + str(frame_num) + ".png"
	dst = draw_save_pt_real+str(color_num)+".png"
	copyfile(src, dst)

