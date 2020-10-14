from skimage.io import imread,imsave
import numpy as np
from utils import *
import pickle
from adversary.shape_homographyMat import Shape
import math

def get_out_pixel_valid(adversary_image,h,frame_num,img):
	h_my_info = prepWarp(adversary_image,h,(img.shape[1],img.shape[0]))
	oidx = h_my_info[0]
	inidx = h_my_info[1]
	canvas_image_croped = img[oidx[1],oidx[0],:]
	r = np.mean(canvas_image_croped[:,0])
	g = np.mean(canvas_image_croped[:,1])
	b = np.mean(canvas_image_croped[:,2])
	return r,g,b
