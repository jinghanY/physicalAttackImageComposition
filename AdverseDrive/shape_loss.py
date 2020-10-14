import numpy as np
import math
import cv2
import os
import imutils
import inspect
import random
from math import sin, cos, radians
import time


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

    def __init__(self, city_name, adversary_name="adversary_" ,transparency=True):
        """
        sizeX and sizeY pertain to the size of the adversary
        on CARLA.
        """
        self.city_name = city_name
        self.sizeX = 400
        #self.sizeX = sizeX;
        self.sizeY = 400
        self.adversary_name = adversary_name

        if transparency:
            self.channels = 4
        else:
            self.channels = 3

        self.clear_canvas()
        self.path = 'adversary/'
        
        self.image_label = self.adversary_name + self.city_name + '.png'
        
        self.episodeNum = 0
        self.data_dir = None 

    def clear_canvas(self):
        self.layer1= np.zeros((self.sizeX, self.sizeY, self.channels), dtype=np.uint8)
        self.layer2= np.zeros((self.sizeX, self.sizeY, self.channels), dtype=np.uint8)
    
    def episode_counter(self):
        self.episodeNum += 1
    
    def draw_lines(self, paras_dict):
    	self.clear_canvas()
    	pts_list = []
    	color_list = []
    	for key in paras_dict:
    		p_dict = paras_dict[key]
    		pts, color = self.line_rotate(p_dict)
    		pts_list.append(pts)
    		color_list.append(color)
    	for i in range(len(pts_list)):
    		pts = pts_list[i]
    		color = color_list[i]
    		cv2.fillPoly(self.layer1,[pts],color)
    	
    	self.canvas=self.layer1[:]
    	cv2.imwrite("{}{}".format(self.path, self.image_label), self.canvas)
    
    
    def line_rotate(self, p_dict,pos = 88):
        """ rot and posX are variables """
        
        # parameters
        rot = p_dict["rot"]
        posY = p_dict["pos"]
        width = p_dict["width"]
        length = p_dict["length"]
        r = p_dict["r"]
        g = p_dict["g"]
        b = p_dict["b"]
        posY = posY + 10
        width = 5 + width
        length = 5 + length

        color=(r,g,b,255)
        
        rot_pos_ori = (posY+width/2,self.sizeX/2)
        
        rt_top_left_ori=(posY,length)
        rt_top_right_ori=(posY+width,length)
        rt_bottom_left_ori=(posY,self.sizeX-length)
        rt_bottom_right_ori=(posY+width,self.sizeX-length)
        
        rt_top_left_rotated_ori = np.array(rotate(rt_top_left_ori, rot, rot_pos_ori))
        rt_top_right_rotated_ori = np.array(rotate(rt_top_right_ori, rot, rot_pos_ori))
        rt_bottom_left_rotated_ori = np.array(rotate(rt_bottom_left_ori, rot, rot_pos_ori))
        rt_bottom_right_rotated_ori = np.array(rotate(rt_bottom_right_ori, rot, rot_pos_ori))
       
        pts = np.array([rt_top_left_rotated_ori,rt_bottom_left_rotated_ori,rt_bottom_right_rotated_ori,rt_top_right_rotated_ori])
        pts = pts.reshape((-1,1,2))
        
        return pts, color
            

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


def calculateDistance(x1,y1,x2,y2):
	dist = math.sqrt((x2-x2)**2 + (y2 - y1)**2)
	return dist

		
def generate_random(number, rectangle_bounds):
	list_of_points = []
	minx, maxx, miny, maxy = rectangle_bounds
	counter = 0
	while counter < number:
		pnt_coordinates = [np.random.randint(minx,maxx), np.random.randint(miny,maxy)]
		# make sure the new points are not close to the existing points
		x2,y2 = pnt_coordinates
		min_dst = 1000
		for point_this in list_of_points:
			x1,y1 = point_this
			dst_this = calculateDistance(x1,y1,x2,y2)
			if dst_this < min_dst:
				min_dst = dst_this

		print(min_dst)
		if pnt_coordinates not in list_of_points and min_dst > 0:
			list_of_points.append(pnt_coordinates)
			counter += 1
		else:
			continue
	list_of_points = np.array(list_of_points)
	list_of_points = list_of_points[list_of_points[:,1].argsort()]
	return list_of_points

def get_rotates(points, rot, rot_pos_ori):
	points_rotated = []
	for i in range(len(points)):
		points_rotated.append(np.array(rotate(points[i], rot, rot_pos_ori)))
	points_rotated = np.array(points_rotated)
	return points_rotated
		
		
	

