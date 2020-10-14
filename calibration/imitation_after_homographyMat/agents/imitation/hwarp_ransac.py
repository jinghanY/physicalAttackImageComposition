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
import prob4 as pb
import sys


def read_init_file(inFile):
	coordinates_init = []
	with open(inFile,"r") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip("\n").split(",")
			line = list(map(int, line))
			coordinates_init.append(line)
	return np.array(coordinates_init)


def read_points_file(inFile):
	coordinates_points = []
	invalid_points = []
	with open(inFile,"r") as f:
		lines = f.readlines()
		ct = 0
		for line in lines:
			line = line.strip("\n").split(" ")
			coord = line[1]
			coord = list(map(int,coord.split(":")[1].split(",")))
			if sum(coord) == 0:
				invalid_points.append(ct)
				ct = ct+1
				continue
			coordinates_points.append(coord)
			ct += 1
	
	return np.array(invalid_points),np.array(coordinates_points)


def splice_original(src,dest,P):
	Hig,Wid= src.shape[0], src.shape[1]
	#xy = np.array([[0,0],[0,Wid-1],[Hig-1,0],[Hig-1,Wid-1]])
	#P = np.concatenate((xy,dpts),axis=1)
	H = pb.getH(P)
	H_inv = linalg.inv(H)
	height,width=dest.shape[0], dest.shape[1]
	for i in range(height):
		for j in range(width):
			xp, yp, wid = H_inv.dot(np.array([j,i,1.]))
			xp, yp = (xp/wid, yp/wid)
			if np.all([xp >= 0, xp<Hig,yp >= 0,yp<Wid]):
				dest[i][j][:] = pb.biliInter(xp,yp,src)[:]
	return dest


def splice(src,dest,H):
	Hig,Wid= src.shape[0], src.shape[1]
	#xy = np.array([[0,0],[0,Wid-1],[Hig-1,0],[Hig-1,Wid-1]])
	#P = np.concatenate((xy,dpts),axis=1)
	H_inv = linalg.inv(H)
	height,width=dest.shape[0], dest.shape[1]
	for i in range(height):
		for j in range(width):
			xp, yp, wid = H_inv.dot(np.array([j,i,1.]))
			xp, yp = (xp/wid, yp/wid)
			if np.all([xp >= 0, xp<Hig,yp >= 0,yp<Wid]):
				dest[i][j][:] = pb.biliInter(xp,yp,src)[:]
	return dest


def geometricDistance(correspondence, h):
	
	p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1),1]))
	estimatep2 = np.dot(h,p1)
	estimatep2 = (1/estimatep2.item(2))*estimatep2

	p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3),1]))
	error = p2 - estimatep2
	return np.linalg.norm(error)




# Runs through ransac algorithm, creating homographies from random correpondences (more than 4). 
def ransac(spts, dpts, thresh):
	assert len(spts == dpts)
	correspondenceList = []
	for i in range(np.shape(pts_src)[0]) :
		(x1, y1) = pts_src[i,:]
		(x2, y2) = pts_dst[i,:]
		correspondenceList.append([x1,y1,x2,y2])
		corr = np.matrix(correspondenceList)
	
	maxInliers = []
	finalH = None
	for i in range(10000):
		idx1 = 0
		idx2 = 1
		idx3 = 2
		idx4 = 3

		four_source = np.array([spts[idx1,:],spts[idx2,:],spts[idx3,:],spts[idx4,:]])
		four_destination = np.array([dpts[idx1,:],dpts[idx2,:],dpts[idx3,:],dpts[idx4,:]])
		p_this = np.concatenate((four_source,four_destination),axis=1)
		
		h = pb.getH(p_this)

		inliers = []

		for j in range(len(corr)):
			d = geometricDistance(corr[j],h) 
			if d<5: # 5 is a hyperparameter
				inliers.append(corr[i])
		if len(inliers) > len(maxInliers):
			maxInliers = inliers
			finalH = h
		sys.stdout.write("Corr size: %d, NumInliers: %d, Max inliers: %d.\n"%(len(corr),len(inliers),len(maxInliers)))
		if len(maxInliers) > (len(corr)*thresh):
			break
	return finalH, maxInliers

## parser
parser = argparse.ArgumentParser()
parser.add_argument("--frame_num","-f",help="frame number")
args = parser.parse_args()
frame_num = int(args.frame_num)

rot_num = 2

out_pt = "../_out/"
hm_pt = "point_coordinates/"


inFile_init = hm_pt+"initial_points.txt"
inFile_points = hm_pt+"points/"+str(frame_num)+".txt"

pts_src = read_init_file(inFile_init)
invalid_point,pts_dst = read_points_file(inFile_points)
if len(invalid_point)!=0:
	pts_src = np.delete(pts_src, invalid_point, 0)

#points_num = 40

#selected_points = random.sample(range(0,len(pts_src)-1),points_num)
#pts_src = np.array([pts_src[i,:] for i in selected_points])
#pts_dst = np.array([pts_dst[i,:] for i in selected_points])
	


pts_src = np.array([pts_src[0,:], pts_src[23,:], pts_src[62,:], pts_src[81,:]])
pts_dst = np.array([pts_dst[0,:], pts_dst[23,:], pts_dst[62,:], pts_dst[81,:]])


xy = pts_src 
dpts = pts_dst
P = np.concatenate((xy,dpts),axis=1)

#print("print P here:")
#print(P)




# run ransac algorithm
estimation_thresh = 0.60
finalH, inliers = ransac(pts_src, pts_dst, estimation_thresh)




# starts from here
from skimage.io import imread, imsave
from os.path import normpath as fn

imgs_path = "point_coordinates/imgs/"+str(frame_num)+"/"
simg = np.float32(imread(fn(imgs_path+"../source.png")))/255.
dimg = np.float32(imread(fn(imgs_path+"real.png")))/255.
simg = simg[:,:,:3]
#simg[10:79,100:111,:] = simg[10:79,100:111,:] + 0.6


#comb = splice_original(simg,dimg,P)
comb = splice(simg,dimg,finalH)
imsave(fn('warped.png'),comb)


## Homography refinement if there are more thatn 4 points.


# end up here


imgs_path = "point_coordinates/imgs/"+str(frame_num)+"/"



h,status = cv2.findHomography(pts_src,pts_dst)
print("h from my method")
print(finalH)
print("h from opencv")
print(h)
#if os.path.exists(imgs_path):
#	shutil.rmtree(imgs_path)
#os.mkdir(imgs_path)


#copyfile(out_pt+"point5/"+str(rot_num)+"/"+str(frame_num)+".png", imgs_path+"real.png")
im_src = cv2.cvtColor(cv2.imread(imgs_path+"../source.png"), cv2.COLOR_BGR2RGB)
#im_dst = cv2.cvtColor(cv2.imread(out_pt+"without_adversary/"+str(rot_num)+"/"+str(frame_num)+".png"), cv2.COLOR_BGR2RGB)
im_dst = cv2.cvtColor(cv2.imread(imgs_path+"real.png"), cv2.COLOR_BGR2RGB)
im_real = cv2.cvtColor(cv2.imread(imgs_path+"real.png"), cv2.COLOR_BGR2RGB)
#im_real = cv2.cvtColor(cv2.imread(out_pt+"all/"+str(rot_num)+"/"+str(frame_num)+".png"), cv2.COLOR_BGR2RGB)

im_src = np.multiply(im_src, 1.0/255.0)
im_dst = np.multiply(im_dst, 1.0/255.0)
im_real = np.multiply(im_real, 1.0/255.0)

im_src[10:79,100:111,:] = im_src[10:79,100:111,:] + 0.6

im_out = cv2.warpPerspective(im_src, finalH, (im_dst.shape[1],im_dst.shape[0]))
im_res = copy.copy(im_dst)

height, width = im_out.shape[0], im_out.shape[1]
for j in range(height):
    for i in range(width):
            if np.max(im_out[j,i,:]) >= 0.5:
                        im_res[j,i,:] = im_out[j,i,:]

plt.imsave(imgs_path+"warped.png", im_res)
