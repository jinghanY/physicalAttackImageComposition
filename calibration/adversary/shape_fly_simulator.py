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

    def __init__(self, city_name, experiment_name, sizeX, sizeY, dataset_pt, transparency=True):
        """
        sizeX and sizeY pertain to the size of the adversary
        on CARLA.
        """
        self.sizeX = 400
        self.sizeY = 400
        self.channels = 4
    
    def clear_canvas(self):
        self.layer1= np.zeros((self.sizeX, self.sizeY, self.channels), dtype=np.uint8)
        self.layer2= np.zeros((self.sizeX, self.sizeY, self.channels), dtype=np.uint8)

    def draw_lines(self, paras_dict):
        self.clear_canvas()
        black_color=(0,0,0,255)

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

        self.canvas=self.layer1[:]
        return self.canvas
    
    def line_rotate(self,p_dict,color=False):
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
        
        if color == "True":
        	color = (r,g,b,0)

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

