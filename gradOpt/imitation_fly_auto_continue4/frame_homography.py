import numpy as np
import argparse
import cv2
#import matplotlib.pyplot as plt
import copy
from shutil import copyfile
import shutil
import os
import random
from scipy import linalg
import prob4 as pb
import math
import sys
from PIL import Image
import re
import glob
from tqdm import tqdm
import prob4 as pb



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
	coordinates = []
	color_values = []
	for x in range(n):
		for y in range(m):
			r,g,b = img_pixel.getpixel((x,y))
			if r > 100 and g == 0 and b == 0:
				color_values.append((r,g,b))
				red_color.append(r)
				coordinates.append([x,y])

	if red_color:
		red_color = np.array(red_color)
		coordinates = np.array(coordinates)
		color_values = np.array(color_values)
		max_index = np.argsort(red_color)[-1]
		color = color_values[max_index]
		x_max, y_max = coordinates[max_index]
		return (x_max, y_max)
	else:
		return (float("nan"), float("nan"))

		
class Frame():
	def __init__(self, frameNum, valid_points, pointsDirSource, pointsDirFrame, angle_range): 
		
		## pointsDirFrame is supposed be to in order. 
		## pointsDir_ are full directory
		## elements in angle_range are folder names instead of actual degree amount.
 
		self.valid_points = np.array(valid_points)
		self.pointsAmount = len(valid_points)
		self.frameNum = frameNum
		self.angle_range = angle_range
		self.angle_range_real = np.array(self.angle_range)-2
		
		## get the coordinates in the source (image).
		self.pointsCoorInSource = {}
		for angle in self.angle_range_real:
			self.pointsCoorInSource[angle] = np.zeros((self.pointsAmount,2))
		self.getPointsCoorInSource(pointsDirSource)
		
		## get the coordinates in the frame (image).
		self.pointsCoorInFrame = {}
		for angle in self.angle_range_real:
			self.pointsCoorInFrame[angle]=np.zeros((self.pointsAmount,2))
		self.getPointsCoorInFrame(pointsDirFrame)
		
		# remove invalid points for each angle
		self.removeInvalidPoints()
		
		## get the homography matrix for this frame.
		self.hs = {}
		for angle in self.angle_range_real:
			self.hs[angle]=None
		#self.getH()

		## get the cv2 homography matrix for this frame.
		self.hs_cv2 = {}
		for angle in self.angle_range_real:
			self.hs_cv2[angle]=None
		self.getH_cv2()

	def getPointsCoorInSource(self,pointsDirSource):
		## update self.pointsCoorInSource
		for angle in self.angle_range:
			pointsDirSource_angle = pointsDirSource+str(angle-1)+".txt"
			initialCoorFull = readInitFile(pointsDirSource_angle)
			# TODO: define valid points which are the points range where you have from the simulator.

			initialCoorFull = initialCoorFull[self.valid_points-1]
			self.pointsCoorInSource[angle-2] = initialCoorFull
		

	def getPointsCoorInFrame(self,pointsDirFrame):
		## update self.pointsCoorInFrame
		print("begin get the coordinate in the frame\n")
		for i in tqdm(range(len(pointsDirFrame))):
			point_dirFrame = pointsDirFrame[i]
			for angle in tqdm(self.angle_range):
				frame_ad = Image.open(point_dirFrame+str(angle)+"/"+str(self.frameNum)+".png")
				coordinate = frame_coordinate(frame_ad)

				self.pointsCoorInFrame[angle-2][i] = coordinate
		print("end up with getting the coordinate in the frame")
	
	def removeInvalidPoints(self):
		for angle in self.angle_range_real:
			assert len(self.pointsCoorInSource[angle]) == len(self.pointsCoorInFrame[angle])
			pts_src = self.pointsCoorInSource[angle]
			pts_dst = self.pointsCoorInFrame[angle]
			# get invalid index
			invalidIndex_angle = []
			for j in range(pts_dst.shape[0]):
				a,b = pts_dst[j,:]
				if math.isnan(a) or math.isnan(b):
					invalidIndex_angle.append(j)

			# apply removing
			pts_src = np.delete(pts_src, invalidIndex_angle,0)
			pts_dst = np.delete(pts_dst, invalidIndex_angle,0)
			self.pointsCoorInSource[angle] = pts_src
			self.pointsCoorInFrame[angle] = pts_dst


	def getH(self):
		## update self.H
		estimation_thresh = 1 # hyperparameter
		for angle in self.angle_range_real:
			finalH, inliers = ransac(self.pointsCoorInSource[angle], self.pointsCoorInFrame[angle], estimation_thresh)
			self.hs[angle] = finalH


	def getH_cv2(self):
		## update self.H
		estimation_thresh = 1 # hyperparameter
		for angle in tqdm(self.angle_range_real):
			h, _ = cv2.findHomography(self.pointsCoorInSource[angle], self.pointsCoorInFrame[angle])
			self.hs_cv2[angle] = h
		print("end up with getting the h matrix")

