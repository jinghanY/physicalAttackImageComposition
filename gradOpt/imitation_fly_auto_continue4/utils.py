import cv2
import numpy as np
import random
import math

def crop_resize_coordinate():
	# objective shape is (200,88)
	shape = (200,88)
	oidx = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
	oidx = np.array(oidx)
	x = oidx[0]
	y = oidx[1]

	oidx = np.stack((oidx[0].flatten(),oidx[1].flatten(),np.ones((oidx.shape[1]*oidx.shape[2]),np.int32)))
	
	# scale up
	# nearest neighbor
	W,H = shape
	image_upscale_size = (800,395)
	W_upscale, H_upscale = image_upscale_size
	scale_x = W_upscale/W
	scale_y = H_upscale/H
	
	x2 = x*scale_x
	y2 = y*scale_y

	# decrop
	y2 += 115
	oidx_scaleup_decrop = np.array([x2,y2])
	oidx_scaleup_decrop = np.stack((oidx_scaleup_decrop[0].flatten(),oidx_scaleup_decrop[1].flatten(),np.ones((oidx_scaleup_decrop.shape[1]*oidx_scaleup_decrop.shape[2]),np.int32)))
	oidx_scaleup_decrop = np.array(oidx_scaleup_decrop)

	return oidx,oidx_scaleup_decrop

def prepWarp_crop_resize_prepCalculate(img,oidx,oidx_scaleup_decrop,h):
	
	inxy = np.matmul(np.linalg.inv(h),np.float32(oidx_scaleup_decrop))
	inidx = np.int32(np.round(inxy[:2,:]/inxy[2:,:]))
	
	mask = np.logical_and(
				np.logical_and(
    				np.logical_and(inidx[0] >= 0,inidx[1] >= 0),
    				np.logical_and(inidx[0] < img.shape[1],inidx[1] < img.shape[0])),inxy[2]>0
    		    )
	
	inidx = inidx[:,mask]
	oidx = oidx[:,mask]
	return [oidx,inidx]

def prepWarp_crop_resize(img,h,shape):
	# Find the coordinate mapping: odix the output coordinate for the canvas in the final image, use inv-h to find inidx: the cooresponding coordinate in the canvas.
	##  The full size of the image is 600x800. Imitation learning model first crop it with img = img [115:510,:]
	##  Basically, cropped the very top and bottom for y-axis. And then it resize the cropped image which has size (395,800) to (88,200).
	
	# input shape is (200,88)
	oidx = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
	oidx = np.array(oidx)
	x = oidx[0]
	y = oidx[1]

	oidx = np.stack((oidx[0].flatten(),oidx[1].flatten(),np.ones((oidx.shape[1]*oidx.shape[2]),np.int32)))
	
	# scale up
	# nearest neighbor
	W,H = shape
	image_upscale_size = (800,395)
	W_upscale, H_upscale = image_upscale_size
	scale_x = W_upscale/W
	scale_y = H_upscale/H
	
	x2 = x*scale_x
	y2 = y*scale_y

	# decrop
	y2 += 115
	oidx_scaleup_decrop = np.array([x2,y2])
	oidx_scaleup_decrop = np.stack((oidx_scaleup_decrop[0].flatten(),oidx_scaleup_decrop[1].flatten(),np.ones((oidx_scaleup_decrop.shape[1]*oidx_scaleup_decrop.shape[2]),np.int32)))
	oidx_scaleup_decrop = np.array(oidx_scaleup_decrop)
	
	inxy = np.matmul(np.linalg.inv(h),np.float32(oidx_scaleup_decrop))
	inidx = np.int32(np.round(inxy[:2,:]/inxy[2:,:]))
	
	mask = np.logical_and(
				np.logical_and(
    				np.logical_and(inidx[0] >= 0,inidx[1] >= 0),
    				np.logical_and(inidx[0] < img.shape[1],inidx[1] < img.shape[0])),inxy[2]>0
    		    )
	
	inidx = inidx[:,mask]
	oidx = oidx[:,mask]
	return [oidx,inidx]

def prepWarp(img,h,shape):
	
	oidx = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
	oidx = np.stack((oidx[0].flatten(),oidx[1].flatten(),np.ones((shape[0]*shape[1]),np.int32)))
	
	inxy = np.matmul(np.linalg.inv(h),np.float32(oidx))
	inidx = np.int32(np.round(inxy[:2,:]/inxy[2:,:]))
	
	mask = np.logical_and(
				np.logical_and(
    				np.logical_and(inidx[0] >= 0,inidx[1] >= 0),
    				np.logical_and(inidx[0] < img.shape[1],inidx[1] < img.shape[0])),inxy[2]>0
    		    )
	
	inidx = inidx[:,mask]
	oidx = oidx[:,mask]
	return [oidx,inidx]

def doWarp(img,hinfo,shape):
	oidx = hinfo[0]
	inidx = hinfo[1]
	out = np.zeros((shape[1],shape[0],img.shape[2]),img.dtype)
	out[oidx[1],oidx[0],:] = img[inidx[1],inidx[0],:]
	return out

def writeFile(imagePoints,points_estimate,distances,outFile_name):
	f = open(outFile_name,"w")
	for i in range(len(imagePoints)):
		imagePoint = imagePoints[i]
		estimatePoint = points_estimate[i]
		distance = distances[i]
		f.write("imagePoint:"+str(imagePoint[0])+","+str(imagePoint[1])+","+"estimatePoint:"+str(estimatePoint[0])+","+str(estimatePoint[1])+",distance_e:"+str(distance))
		f.write("\n")
		
def calculateDistance(x1,y1,x2,y2):
	dist = math.sqrt((x2-x1)**2 + (y2 - y1)**2)
	return dist

def pointPerspectiveTrans(point,h):
	point = np.array(point)
	pts_transed = []
	x,y=point
	p = np.array((x,y,1)).reshape((3,1))
	temp_p = h.dot(p)
	sum_this = np.sum(temp_p ,1)
	px = int(round(sum_this[0]/sum_this[2]))
	py = int(round(sum_this[1]/sum_this[2]))
	pts_transed.append([px,py])
	pts_transed = np.array(pts_transed)
	return np.int32([pts_transed]) 

def randomNum(min_no,max_no,no_points,random_seed):
	random.seed(random_seed)
	return [random.randint(min_no, max_no) for j in range(no_points)]
	
