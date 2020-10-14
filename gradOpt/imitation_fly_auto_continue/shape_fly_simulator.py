import numpy as np
import math
import cv2
import os
import imutils
import inspect
import random
from math import sin, cos, radians
import time
import scipy
from skimage.io import imread,imsave
import copy

class Shape:
    """
    Library containing different shapes, along with the ability
    to create .png images with the shapes.
    """

    def __init__(self, city_name, experiment_name, sizeX, sizeY, transparency=True):
        """
        sizeX and sizeY pertain to the size of the adversary
        on CARLA.
        """
        self.sizeX = 400
        self.sizeY = 400;
    
    def draw_line(self, p_dict):
    	layer1= np.zeros((self.sizeX, self.sizeY, 4), dtype=np.uint8)
    	pts_list = []
    	color_list = []
    	pts, color = self.line_rotate(p_dict)
    	pts_list.append(pts)
    	color_list.append(color)
    	cv2.fillPoly(layer1,[pts],color)
    	return layer1
    
    def draw_lines(self, paras_dict):
    	layer1= np.zeros((self.sizeX, self.sizeY, 4), dtype=np.uint8)
    	pts_list = []
    	color_list = []
    	for key in paras_dict:
    		p_dict = paras_dict[key]
    		pts, color = self.line_rotate(p_dict)
    		pts_list.append(pts)
    		color_list.append(color)
    		cv2.fillPoly(layer1,[pts],color)
    	return layer1
    
    def line_rotate(self, p_dict):
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
        
        rt_top_left_rotated_ori = rotate(rt_top_left_ori, rot, rot_pos_ori)
        rt_top_right_rotated_ori = rotate(rt_top_right_ori, rot, rot_pos_ori)
        rt_bottom_left_rotated_ori = rotate(rt_bottom_left_ori, rot, rot_pos_ori)
        rt_bottom_right_rotated_ori = rotate(rt_bottom_right_ori, rot, rot_pos_ori)

        pts = np.array([rt_top_left_rotated_ori,rt_bottom_left_rotated_ori,rt_bottom_right_rotated_ori,rt_top_right_rotated_ori])
        return pts, color
    
def rotate(point,angle,center_point):
	angle_rad = radians(angle % 360)
	new_point = (point[0]-center_point[0], point[1]-center_point[1])
	new_point = (new_point[0]*cos(angle_rad)-new_point[1]*sin(angle_rad), new_point[0]*sin(angle_rad)+new_point[1]*cos(angle_rad))
	new_point = [int(new_point[0]+center_point[0]), int(new_point[1]+center_point[1])]
	return new_point
