import numpy as np
import math
import cv2
import os
import imutils
import inspect
import random
from math import sin, cos, radians
import time
from skimage.io import imread,imsave

def random_n_offsets(random_seed, range_chose, num):
    l = range_chose[1] - range_chose[0]
    try:
    	random.seed(random_seed)
    	offsets = random.sample(range(range_chose[0], range_chose[1]), num)
    except:
    	random.seed(random_seed)
    	offsets = random.sample(range(range_chose[0], range_chose[1]),l-2)
    
    return offsets

class Shape:
    """
    Library containing different shapes, along with the ability
    to create .png images with the shapes.
    """

    def __init__(self, city_name, radius,sizeX=400, sizeY=400, adversary_name="adversary_", transparency=True):
        """
        sizeX and sizeY pertain to the size of the adversary
        on CARLA.
        Or adversary_name="adversarybeta_"
        """
        self.city_name = city_name
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.adversary_name = adversary_name
        self.radius = radius

        if transparency:
            self.channels = 4
        else:
            self.channels = 3

        self.clear_canvas()
        self.path = 'adversary/'
        self.image_label = self.adversary_name + self.city_name + '.png'

    def clear_canvas(self):
        self.layer1= np.zeros((self.sizeX, self.sizeY, self.channels), dtype=np.uint8)
        self.layer2= np.zeros((self.sizeX, self.sizeY, self.channels), dtype=np.uint8)

    def lines_rotate(self,opt,color_option=False):
        """ rot and posX are variables """
        self.clear_canvas()

        color_input_file = self.path + "color.txt"
        color_input = load_color(color_input_file)

        points_path = self.path + "initial_points"
        points = load_points(points_path)
        
        red_color=[255,0,0,255]
        black_color=(0,0,0,255)

        # transparent
        if color_option == "True":
        	red_color=[255,0,0,0]
        	black_color=(0,0,0,0)
        # horizental size is Y and the vertical size is X
        posY = 0
        rot = 0
        rot_1 = rot
        rot_2 = rot

        width = 200
        length = 0
        distance = 0

        rot_pos_ori = (posY+width+distance/2,self.sizeX/2)
        rt_top_left_ori=(posY,length)
        rt_top_right_ori=(posY+width,length)
        rt_bottom_left_ori=(posY,self.sizeX-length)
        rt_bottom_right_ori=(posY+width,self.sizeX-length)

        posY_2 = posY + width + distance
        rt_top_left_ori_2=(posY_2,length)
        rt_top_right_ori_2=(posY_2+width,length)
        rt_bottom_left_ori_2=(posY_2,self.sizeX-length)
        rt_bottom_right_ori_2=(posY_2+width,self.sizeX-length)

        rt_top_left_rotated_ori = np.array(rotate(rt_top_left_ori, rot_1, rot_pos_ori))
        rt_top_right_rotated_ori = np.array(rotate(rt_top_right_ori, rot_1, rot_pos_ori))
        rt_bottom_left_rotated_ori = np.array(rotate(rt_bottom_left_ori, rot_1, rot_pos_ori))
        rt_bottom_right_rotated_ori = np.array(rotate(rt_bottom_right_ori, rot_1, rot_pos_ori))
        rt_top_left_rotated_ori_2 = np.array(rotate(rt_top_left_ori_2, rot_2, rot_pos_ori))
        rt_top_right_rotated_ori_2 = np.array(rotate(rt_top_right_ori_2, rot_2, rot_pos_ori))
        rt_bottom_left_rotated_ori_2 = np.array(rotate(rt_bottom_left_ori_2, rot_2, rot_pos_ori))
        rt_bottom_right_rotated_ori_2 = np.array(rotate(rt_bottom_right_ori_2, rot_2, rot_pos_ori))

        pts = np.array([rt_top_left_rotated_ori,rt_bottom_left_rotated_ori,rt_bottom_right_rotated_ori,rt_top_right_rotated_ori])
        pts_2 = np.array([rt_top_left_rotated_ori_2,rt_bottom_left_rotated_ori_2,rt_bottom_right_rotated_ori_2,rt_top_right_rotated_ori_2])

        pts = pts.reshape((-1,1,2))
        pts_2 = pts_2.reshape((-1,1,2))
        cv2.fillPoly(self.layer1,[pts],black_color)
        cv2.fillPoly(self.layer1,[pts_2],black_color)

        ## sample points for homography
        #rectangle_bounds = (10,390,10,390)
        #points = generate_random(60,rectangle_bounds)
        #write_16_points(points,self.path+"initial_points")
        
        # read points in
        
        self.radius = 4
        start_point = (opt-1)*5
        points = points[start_point:start_point+5]
        for i in range(len(points)):
        	color_this = list(color_input[i])
        	r,g,b = color_this
        	red_color = (int(r),int(g),int(b),255)
        	cv2.circle(self.layer2,tuple(points[i]),self.radius,red_color,-1)
        self.canvas=self.layer1[:]
        cnd=self.layer2[:,:,3]>0
        self.canvas[cnd]=self.layer2[cnd]
        imsave(self.path+self.image_label,self.canvas)

def rotate(point,angle,center_point):
	angle_rad = radians(angle % 360)
	new_point = (point[0]-center_point[0], point[1]-center_point[1])
	new_point = (new_point[0]*cos(angle_rad)-new_point[1]*sin(angle_rad), new_point[0]*sin(angle_rad)+new_point[1]*cos(angle_rad))
	new_point = (int(new_point[0]+center_point[0]), int(new_point[1]+center_point[1]))
	return new_point


def write_16_points(points, fileName):
	#points = np.array(points)
	#points = points[points[:,1].argsort()]
	with open(fileName, "w") as f:
		for i in range(len(points)):
			point = points[i]
			f.write(str(point[0])+","+str(point[1])+"\n")

def load_points(fileName):
	points = []
	with open(fileName, "r") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip("\n").split(",")
			line = list(map(int, line))
			points.append(np.array(line))
	return np.array(points)

def load_color(fileName):
	points = []
	with open(fileName, "r") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip("\n").split(",")
			line = list(map(int, line))
			points.append(np.array(line))
	return np.array(points)

def calculateDistance(x1,y1,x2,y2):
	dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
	return dist

		
def generate_random(number, rectangle_bounds):
	list_of_points = []
	minx, maxx, miny, maxy = rectangle_bounds
	counter = 1
	while counter <= number:
		pnt_coordinates = [np.random.randint(minx,maxx), np.random.randint(miny,maxy)]
		#pnt_coordinates = [int(random.uniform(minx, maxx)), int(random.uniform(miny, maxy))]
		# make sure the new points are not close to the existing points
		x2,y2 = pnt_coordinates
		min_dst = 1000
		for point_this in list_of_points:
			x1,y1 = point_this
			#print(x1,y1) 
			dst_this = calculateDistance(x1,y1,x2,y2)
			if dst_this < min_dst:
				min_dst = dst_this

		if pnt_coordinates not in list_of_points and min_dst >= 44:
			list_of_points.append(pnt_coordinates)
			print(len(list_of_points))
			counter += 1
		else:
			continue
	list_of_points = np.array(list_of_points)
	#list_of_points = list_of_points[list_of_points[:,1].argsort()]
	return list_of_points


def get_rotates(points, rot, rot_pos_ori):
	points_rotated = []
	for i in range(len(points)):
		points_rotated.append(np.array(rotate(points[i], rot, rot_pos_ori)))
	points_rotated = np.array(points_rotated)
	return points_rotated
