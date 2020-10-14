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

class Shape:
    """
    Library containing different shapes, along with the ability
    to create .png images with the shapes.
    """

    def __init__(self, city_name, radius=1,sizeX=400, sizeY=400, adversary_name="adversary_", transparency=True):
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

    def lines_rotate(self,color_input,color_option=False):
        """ rot and posX are variables """
        self.clear_canvas()

        r,g,b = color_input
        black_color = (int(r),int(g),int(b),255)
        print("check here")
        print(black_color)
        
        posY = 0
        rot = 0
        rot_1 = rot
        rot_2 = rot

        width = 400
        length = 0
        distance = 0

        rt_top_left_ori=(posY,length)
        rt_top_right_ori=(posY+width,length)
        rt_bottom_left_ori=(posY,self.sizeX-length)
        rt_bottom_right_ori=(posY+width,self.sizeX-length)

        pts = np.array([rt_top_left_ori,rt_bottom_left_ori,rt_bottom_right_ori,rt_top_right_ori])

        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(self.layer1,[pts],black_color)

        ## sample points for homography
        #rectangle_bounds = (0,400,0,400)
        #colors_input = generate_random(100)
        #write_16_points(colors_input,self.path+"color.txt")
        
        self.canvas=self.layer1[:]
        cnd=self.layer2[:,:,3]>0
        self.canvas[cnd]=self.layer2[cnd]
        imsave(self.path+self.image_label,self.canvas)


def write_16_points(points, fileName):
	#points = np.array(points)
	#points = points[points[:,1].argsort()]
	with open(fileName, "w") as f:
		for i in range(len(points)):
			point = points[i]
			f.write(str(point[0])+","+str(point[1])+","+str(point[2])+"\n")

def load_color(fileName):
	points = []
	with open(fileName, "r") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip("\n").split(",")
			line = list(map(int, line))
			points.append(np.array(line))
	return np.array(points)


def generate_random(number):
	list_of_colors = []
	counter = 1
	while counter <= number:
		color = [np.random.randint(0,255), np.random.randint(0,255),np.random.randint(0,255)]

		if color not in list_of_colors: 
			list_of_colors.append(color)
			#print(len(list_of_colors))
			counter += 1
		else:
			continue
	list_of_colors = np.array(list_of_colors)
	return list_of_colors
