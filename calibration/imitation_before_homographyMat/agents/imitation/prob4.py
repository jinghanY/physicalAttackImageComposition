## Default modules imported. Import more if you need to.
from scipy import linalg
import numpy as np


## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts):
	#x=pts[:,0]
	A = []
	for point in pts:
		point1 = np.matrix([point.item(0), point.item(1), 1])
		point2 = np.matrix([point.item(2), point.item(3), 1])
		A_row1 = [-point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2), 0, 0, 0,point2.item(0) * point1.item(0), point2.item(0) * point1.item(1), point2.item(0) * point1.item(2)]
		A_row2 = [0, 0, 0, -point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2),point2.item(1) * point1.item(0), point2.item(1) * point1.item(1), point2.item(1) * point1.item(2)]
		
		A.append(A_row1)
		A.append(A_row2)
		#A.append(A_row3)
	A = np.matrix(A)
	U, S, V = np.linalg.svd(A)
	H = np.reshape(V[8],(3,3))
	H = (1/H.item(8))*H
	return H
    

# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.

def validPix(img,x,y):
	return img[min(max(0,x),img.shape[0]-1), min(max(y,0), img.shape[1]-1)]

def biliInter(x,y,image):
	x1,x2 = int(np.floor(x)), int(np.ceil(x))
	y1,y2 = int(np.floor(y)), int(np.ceil(y))
	Q11 = validPix(image,x1,y1)
	Q12 = validPix(image,x1,y2)
	Q21 = validPix(image,x2,y1)
	Q22 = validPix(image,x2,y2)
	if x1 == x2:
		R1 = Q11
		R2 = Q12 
	else:
		R1 = ((x2-x)/(x2-x1))*Q11+((x-x1)/(x2-x1))*Q21
		R2 = ((x2-x)/(x2-x1))*Q12 + ((x-x1)/(x2-x1))*Q22

	if y1 == y2:
		return R1
	else:
		P = ((y2-y)/(y2-y1))*R1 + ((y-y1)/(y2-y1))*R2
		return P

def splice(src,dest,dpts):
	Hig,Wid= src.shape[0], src.shape[1]
	xy = np.array([[0,0],[0,Wid-1],[Hig-1,0],[Hig-1,Wid-1]])
	P = np.concatenate((xy,dpts),axis=1)
	H = getH(P)
	H_inv = linalg.inv(H)
	height,width=dest.shape[0], dest.shape[1]
	for i in range(height):
		for j in range(width):
			xp, yp, wid = H_inv.dot(np.array([j,i,1.]))
			xp, yp = (xp/wid, yp/wid)
			if np.all([xp >= 0, xp<Hig,yp >= 0,yp<Wid]):
				dest[i][j][:] = biliInter(xp,yp,src)[:]
	return dest
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


#simg = np.float32(imread(fn('inputs/p4src.png')))/255.
#dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
#dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

#comb = splice(simg,dimg,dpts)

#imsave(fn('outputs/prob4.png'),comb)
