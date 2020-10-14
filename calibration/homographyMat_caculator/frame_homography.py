import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import copy
from shutil import copyfile
import shutil
import os
import random
from scipy import linalg
import homographyMat_caculator.homographyMatMy as pb
import math
import sys
from PIL import Image
import re
import glob
from tqdm import tqdm
import pickle


def geometricDistance(correspondence,h):
	p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1),1]))
	estimatep2 = np.dot(h,p1)
	estimatep2 = (1/estimatep2.item(2))*estimatep2

	p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3),1]))
	error = p2 - estimatep2
	return np.linalg.norm(error)


def ransac(spts, dpts, thresh):
	assert len(spts) == len(dpts)
	correspondenceList = []
	for i in range(np.shape(spts)[0]):
		(x1, y1) = spts[i,:]
		(x2, y2) = dpts[i,:]
		correspondenceList.append([x1,y1,x2,y2])
		corr = np.matrix(correspondenceList)

	maxInliers = []
	finalH = None
	for i in range(1000):
		(idx1, idx2, idx3, idx4) = random.sample(range(len(spts)),4)
		four_source = np.array([spts[idx1,:],spts[idx2,:],spts[idx3,:],spts[idx4,:]])
		four_destination = np.array([dpts[idx1,:],dpts[idx2,:],dpts[idx3,:],dpts[idx4,:]])
		p_this = np.concatenate((four_source,four_destination),axis=1)
		h = pb.getH(p_this)
		inliers = []
		for j in range(len(corr)):
			d = geometricDistance(corr[j],h)
			if d<5:
				inliers.append(corr[j])
		if len(inliers) > len(maxInliers):
			maxInliers = inliers
			finalH = h
		sys.stdout.write("Corr size: %d, NumInliers: %d, Max inliers: %d.\n"%(len(corr),len(inliers),len(maxInliers)))
		if len(maxInliers) > (len(corr)*thresh):
			break
	return finalH, maxInliers
		

def readInitFile(inFile):
	coordinates_init = []
	with open(inFile,"r") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip("\n").split(",")
			line = list(map(int, line))
			coordinates_init.append(line)
	return np.array(coordinates_init)


def frame_coordinate(img):
	# Input: is a particular frame
	# Return:
	#    coordinate of the point
	#    None: if there no sample in this frame
	img_pixel = img.convert('RGB')
	n,m = img_pixel.size

	red_color = []
	coordinates1 = []
	color_values1 = []
	coordinates2 = []
	color_values2 = []
	coordinates3 = []
	color_values3 = []
	coordinates4 = []
	color_values4 = []
	coordinates5 = []
	color_values5 = []
	coordinates6 = []
	color_values6 = []
	
	for x in range(n):
		for y in range(m):
			r,g,b = img_pixel.getpixel((x,y))
			if r > 100 and g == 0 and b == 0:
				color_values1.append((r,g,b))
				coordinates1.append([x,y])
			elif r ==0 and g > 100 and b ==0:
				color_values2.append((r,g,b))
				coordinates2.append([x,y])
			elif r ==0 and g ==0 and b >100:
				color_values3.append((r,g,b))
				coordinates3.append([x,y])
			elif r ==0 and g > 100 and b >100:
				color_values4.append((r,g,b))
				coordinates4.append([x,y])
			elif r > 100 and g == 0 and b >100:
				color_values5.append((r,g,b))
				coordinates5.append([x,y])
				
	res_coordinates = []
	if color_values1:	
		color_values1 = np.array(color_values1)
		coordinates1 = np.array(coordinates1)
		color_value1 = color_values1[:,0]
		max_index = np.argsort(color_value1)[-1]
		color_value1 = color_values1[max_index]
		x_max, y_max = coordinates1[max_index]
		res_coordinates.append((x_max,y_max))
	else:
		res_coordinates.append((float("nan"),float("nan")))
	
	if color_values2:	
		color_values2 = np.array(color_values2)
		coordinates2 = np.array(coordinates2)
		color_value2 = color_values2[:,1]
		max_index = np.argsort(color_value2)[-1]
		color_value2 = color_values2[max_index]
		x_max, y_max = coordinates2[max_index]
		res_coordinates.append((x_max,y_max))
	else:
		res_coordinates.append((float("nan"),float("nan")))
	
	if color_values3:	
		color_values3 = np.array(color_values3)
		coordinates3 = np.array(coordinates3)
		color_value3 = color_values3[:,2]
		max_index = np.argsort(color_value3)[-1]
		color_value3 = color_values3[max_index]
		x_max, y_max = coordinates3[max_index]
		res_coordinates.append((x_max,y_max))
	else:
		res_coordinates.append((float("nan"),float("nan")))
	
	
	if color_values4:	
		color_values4 = np.array(color_values4)
		coordinates4 = np.array(coordinates4)
		color_value4 = color_values4[:,2]
		max_index = np.argsort(color_value4)[-1]
		color_value4 = color_values4[max_index]
		x_max, y_max = coordinates4[max_index]
		res_coordinates.append((x_max,y_max))
	else:
		res_coordinates.append((float("nan"),float("nan")))
	
	if color_values5:	
		color_values5 = np.array(color_values5)
		coordinates5 = np.array(coordinates5)
		color_value5 = color_values5[:,0]
		max_index = np.argsort(color_value5)[-1]
		color_value5 = color_values5[max_index]
		x_max, y_max = coordinates5[max_index]
		res_coordinates.append((x_max,y_max))
	else:
		res_coordinates.append((float("nan"),float("nan")))
	
	return res_coordinates
	
		
class Frame():
	def __init__(self, frameNum, valid_points, pointsDirSource, pointsDirFrame_radius1): 
		
		### pointsDirFrame is supposed be to in order. 
		### pointsDir_ are full directory
 
		self.valid_points = np.array(valid_points)
		self.pointsAmount = len(self.valid_points)
		self.frameNum = frameNum
		
		### get the coordinates in the source (image).
		self.pointsCoorInSource = np.zeros((self.pointsAmount,2))
		self.getPointsCoorInSource(pointsDirSource)
		
		### get the coordinates in the frame (image).
		self.pointsCoorInFrame=[]
		self.getPointsCoorInFrame(pointsDirFrame_radius1)

		## remove invalid points for each angle
		self.removeInvalidPoints()

		## get the homography using our implementation
		#self.hs = {}
		#self.hs["h"]=None
		#try:
		#	self.getH()
		#except:
		#	pass
		
		## get the cv2 homography matrix for this frame.
		self.hs_cv2 = {}
		self.hs_cv2["h"]=None
		self.getH_cv2()

	def getPointsCoorInSource(self,pointsDirSource):
		pointsDirSource = "adversary/initial_points"
		initialCoorFull = readInitFile(pointsDirSource)
		self.pointsCoorInSource = initialCoorFull
		
	def getPointsCoorInFrame(self,pointsDirFrame_radius1):
		## update self.pointsCoorInFrame
		print("begin get the coordinate in the frame\n")
		self.pointsCoorInFrame = []
		for i in range(len(pointsDirFrame_radius1)):
			point_dirFrame_radius1 = pointsDirFrame_radius1[i]
			frame_ad_radius1 = Image.open(point_dirFrame_radius1+str(self.frameNum)+".png")
			coordinate = frame_coordinate(frame_ad_radius1)
			self.pointsCoorInFrame += coordinate
		self.pointsCoorInFrame = np.array(self.pointsCoorInFrame)
		print("end up with getting the coordinate in the frame")
	
	def removeInvalidPoints(self):
		
		
		pts_src = self.pointsCoorInSource
		pts_dst = self.pointsCoorInFrame
		assert len(self.pointsCoorInSource) == len(self.pointsCoorInFrame)
		invalidIndex = []
		self.NumPoints = []
		for j in range(pts_dst.shape[0]):
			a,b = pts_dst[j,:]
			if math.isnan(a) or math.isnan(b):
				invalidIndex.append(j)
			else:
				self.NumPoints.append(j)

		# apply removing
		print(pts_src)
		print(pts_dst)
		pts_src = np.delete(pts_src, invalidIndex,0)
		pts_dst = np.delete(pts_dst, invalidIndex,0)
		print(pts_src)
		print(pts_dst)
		self.pointsCoorInSource = pts_src
		self.pointsCoorInFrame = pts_dst

	def invisibleCheck(self,h):
		x = self.pointsCoorInSource[:,0]
		y = self.pointsCoorInSource[:,1]
		canvas_idx = np.meshgrid(x,y)
		canvas_idx = np.array(canvas_idx)
		canvas_idx = np.stack((canvas_idx[0].flatten(),canvas_idx[1].flatten(),np.ones((canvas_idx.shape[1]*canvas_idx.shape[2]),np.int32)))
		out_coord = np.matmul(h,np.float32(canvas_idx))
		print(np.mean(np.sign(out_coord[2,:])))
		if np.mean(np.sign(out_coord[2,:])) < 0:
			h = -h
		return h
	
	def getH(self):
		## update self.H
		estimation_thresh = 1 # hyperparameter
		finalH, inliers = ransac(self.pointsCoorInSource, self.pointsCoorInFrame, estimation_thresh)
		
		finalH = self.invisibleCheck(finalH)
		self.hs["h"] = finalH

	def getH_cv2(self):
		## update self.H
		h, _ = cv2.findHomography(self.pointsCoorInSource, self.pointsCoorInFrame)
		if type(h).__module__ == np.__name__:
			if h.any() == None:
				pass
			else:
				h = self.invisibleCheck(h)
		else:
			if h == None:
				pass
			else:
				h = self.invisibleCheck(h)
		print(h)
		self.hs_cv2["h"] = h

