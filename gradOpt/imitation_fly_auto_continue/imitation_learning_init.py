from __future__ import print_function
from skimage.io import imread, imsave
import os

import scipy

import tensorflow as tf
import numpy as np
from skimage.io import imread, imsave
import cv2
import time
import sys
import itertools
import copy

slim = tf.contrib.slim

from copy import deepcopy
from imitation_learning_network import load_imitation_learning_network
import math
import random
import json
import pickle
from PIL import Image
from configparser import ConfigParser
import shutil
from collections import OrderedDict
import glob
import re
from utils import *


"""custom libraries"""
from shape_fly_simulator import Shape

def readFile(fileName):
	f = open(fileName,"r")
	lines = f.readlines()
	parameters = []
	losses = []
	iterations = []
	iteration_no = 0
	for line_idx in range(len(lines)):
		line = lines[line_idx]
		if line.startswith("rot="):
			line = line.split("rot")
			line = line[1:]
			adversary_paras = []
			try:
				for i in range(len(line)):
					subline = line[i]
					subline = subline.strip(",").split(",")
					adversary = subline
					if i == len(line)-1:
						adversary = adversary[:-2]
					adversary_paras_this = [float(x.split("=")[-1]) for x in adversary]
					adversary_paras.append(adversary_paras_this)
				loss = float(line[-1].strip("\n").split("loss=")[-1])
				losses.append(loss)
			except:
				continue
			parameters.append(adversary_paras)
			# get the iteration number for this set
			line_iteration = lines[line_idx+1]
			iteration_this_string = line_iteration.split("=")[-1]
			iteration_this_string = re.findall(r'\d+',iteration_this_string)[0]
			try:
				iteration_no = int(re.match(r'^(0*)([^0].*)$', iteration_this_string).group(2))
			except:
				iteration_no = 0
	iterations.append(iteration_no)

	return parameters, losses, iterations

def parameter_one_randomGen(x,y,random_seed_one_randomGen):
	"""
	Generate a random number between x and y
	precision 0.1 
	"""
	random.seed(random_seed_one_randomGen)
	res = random.uniform(x, y)
	return math.floor(res * 10) / 10

def parameter_one_randomGen_color(x,y,random_seed_one_randomGen):
	"""
	Generate a random number between x and y
	precision 0.1 
	"""
	random.seed(random_seed_one_randomGen)
	res = random.uniform(x, y)
	return math.floor(res * 10) / 10

def para_gen(para_name,random_seed_para_gen,step_size):
	if para_name.startswith("rot"):
		return [parameter_one_randomGen(20,180,random_seed_para_gen),180.0,step_size]
	elif para_name.startswith("pos"):
		return [parameter_one_randomGen(0,380,random_seed_para_gen),380.0,step_size]
	elif para_name.startswith("width"): 
		return [parameter_one_randomGen(20,100,random_seed_para_gen),100.0,step_size]
	elif para_name.startswith("length"):
		return [parameter_one_randomGen(0,70,random_seed_para_gen),70.0,step_size]

def dotR(inputStr):
	return str(inputStr).replace(".","-")
    	
def create_init_paras(paras_name,random_seed_init_paras,step_size):
	random.seed(random_seed_init_paras)
	random_seeds = random.sample(range(1, 1000), len(paras_name))
	paras_d = {}
	paras_d["name"] = []
	paras_d["value"] = []
	paras_d["norm_factor"] = []
	paras_d["step_factor"] = []

	for i in range(len(paras_name)):
		para_name = paras_name[i]
		random_seed_this = random_seeds[i]
		init_value,norm_factor,step_factor = para_gen(para_name,random_seed_this,step_size)
		paras_d["name"].append(para_name)
		paras_d["value"].append(float(init_value/norm_factor))
		paras_d["norm_factor"].append(float(norm_factor))
		paras_d["step_factor"].append(float(step_factor))
	return paras_d 

def para_gen_color(para_name,random_seed_para_gen):
	if para_name.startswith("r"):
		return parameter_one_randomGen_color(0,255,random_seed_para_gen)
	elif para_name.startswith("g"):
		return parameter_one_randomGen(0,255,random_seed_para_gen)
	elif para_name.startswith("b"):
		return parameter_one_randomGen(0,255,random_seed_para_gen)

def create_init_color(paras_name,random_seed_init_paras):
	random.seed(random_seed_init_paras)
	random_seeds = random.sample(range(1, 1000), len(paras_name)+4)
	random_seeds = random_seeds[4:]
	paras_d = {}
	paras_d["name"] = []
	paras_d["value"] = []

	for i in range(len(paras_name)):
		para_name = paras_name[i]
		random_seed_this = random_seeds[i]
		init_value = para_gen_color(para_name,random_seed_this)
		paras_d["name"].append(para_name)
		paras_d["value"].append(float(init_value)/255.)
	
	return paras_d 

def create_file_name(parameters,image_pt,frame_count):
	file_name = image_pt + str(frame_count)+"/"
	for i in range(len(parameters)):
		file_name = file_name + dotR(parameters[i]) + "_"
	file_name = file_name+".png"
	return file_name

def tf_rad2deg(rad):
	pi_on_180 = 0.017453292519943295
	return rad / pi_on_180

class ImitationLearningInit():

    def __init__(self, paras_dict, color_dict,epoch, random_seed, step_size, memory_fraction=0.25, image_cut=[115, 510],LR=0.1,task='right-turn/',intersection="42_47/"):

        self.epoch = epoch
        self.step_size = step_size
        self.paras_ds = OrderedDict()
        self.task = task
        self.intersection = intersection
        self.trajectories = [1,2,3,4]
        random_seed = random_seed
        random.seed(random_seed)
        random_seeds = random.sample(range(1,1000),len(paras_dict))
        random_seeds = random_seeds[::-1]
        ct_random_seed = 0
        self.color_grads_coordinates = []
        self.adversary_num = len(paras_dict)
        
        for adversary_name in paras_dict:
        	random_seed_this = random_seeds[ct_random_seed]
        	self.paras_d = create_init_paras(paras_dict[adversary_name],random_seed_this,self.step_size)
        	self.paras_ds[adversary_name] = self.paras_d
        	ct_random_seed += 1
        
        # continue the experiments
        para_default = [[0,88,10,40]]
        
        self.iter = 0
        self.parameters_previous_unnormed = None
        
        self.name = OrderedDict()
        self.parameters_init = []
        self.norm_factors = []
        self.step_factors = []
        for adversary_name in self.paras_ds:
        	paras_d_this = self.paras_ds[adversary_name]
        	self.name[adversary_name] = paras_d_this["name"]
        	self.parameters_init += paras_d_this["value"]
        	self.norm_factors += paras_d_this["norm_factor"]
        	self.step_factors += paras_d_this["step_factor"]

        self.intervals = []
        for i in range(len(self.norm_factors)):
        	self.intervals.append(self.step_factors[i]/self.norm_factors[i])
        
        self.LR = LR 
        self.LR_color = LR
        
        # get color parameters
        ## initialization
        ct_random_seed = 0
        self.color_ds = OrderedDict()
        for adversary_name in color_dict:
        	random_seed_this = random_seeds[ct_random_seed]
        	self.color_d = create_init_color(color_dict[adversary_name],random_seed_this)
        	self.color_ds[adversary_name] = self.color_d
        	ct_random_seed += 1
        
        self.colors_name = OrderedDict()
        self.colors_init = []
        for adversary_name in self.color_ds:
        	colors_d_this = self.color_ds[adversary_name]
        	self.colors_name[adversary_name] = colors_d_this["name"]
        	self.colors_init += colors_d_this["value"]
        
        self.dataset_rt_clean = "../calibration/"
        self.dataset_rt_noises = "../calibration/" 

        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.visible_device_list = '0'
        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        import os
        tf_device = '/cpu:0'

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        self._sess = tf.Session(config=session_conf)

        self._image_size = (88, 200, 3)

        # get frame number
        import os
        self.get_hm_cleanFrames()

        self._sess = tf.Session(config=config_gpu)
        self.draw =  Shape("Town01_nemesisA",'single-line',200,200)

        # modify run_CIL to define init_angle_width
        self.adversary_parameters = []
        self.color_parameters = []
        self.weights_parameters = []
        self.adversary_parameters_blur = []

        self.image_inputs_floor = {}
        self.image_inputs_ceil = {}

        with tf.device(tf_device):
            for i in range(len(self.parameters_init)):
            	var = tf.Variable(initial_value=self.parameters_init[i], name="adversary_"+str(i), dtype="float32",constraint=lambda t: tf.clip_by_value(t, 0, 1))
            	self.adversary_parameters.append(var)
            
            for i in range(len(self.colors_init)):
            	var = tf.Variable(initial_value=self.colors_init[i], name="color_"+str(i), dtype="float32",constraint=lambda t: tf.clip_by_value(t, 0, 1))
            	self.color_parameters.append(var)
            
            for i in range(len(self.adversary_parameters)):
            	adversary_parameter = self.adversary_parameters[i]
            	adversary_parameter1 = tf.floor(adversary_parameter/self.intervals[i])*self.intervals[i]
            	adversary_parameter2 = self.intervals[i] + adversary_parameter1 
            	weight1 = (adversary_parameter2 - adversary_parameter)/self.intervals[i]
            	weight2 = 1 - weight1
            	self.weights_parameters.append([weight1, weight2])
            	self.adversary_parameters_blur.append([adversary_parameter1,adversary_parameter2])

            self.image_inputs_actual = tf.placeholder(dtype="float32",shape=[self.batchsize,self._image_size[0],
            																	   self._image_size[1],
            																	   self._image_size[2]],
            																	   name = "I_actual")
            
            self.coordinates_placeholder = tf.placeholder(dtype="float32",shape=[self.adversary_num,self.batchsize,self._image_size[0],
            																	   self._image_size[1]],
            																	   name = "coordinates")
            
            for i in range(len(self.parameters_init)):
            	self.image_inputs_floor[i] = tf.placeholder(dtype="float32",shape=[self.batchsize,self._image_size[0],
            																	   self._image_size[1],
            																	   self._image_size[2]],
            																	   name = "I_floor_"+str(i))
            
            	self.image_inputs_ceil[i] = tf.placeholder(dtype="float32",shape=[self.batchsize,self._image_size[0],
            																	   self._image_size[1],
            																	   self._image_size[2]],
            																	   name = "I_ceil_"+str(i))
            
            self.images_blur = []
            self._input_images = self.image_inputs_actual 
            # blur verision
            for i in range(len(self.parameters_init)):
            	weight1, weight2 = self.weights_parameters[i]
            	image_blur = weight1*self.image_inputs_floor[i] + weight2*self.image_inputs_ceil[i]
            	self.images_blur.append(image_blur)

            for i in range(len(self.images_blur)):
            	self._input_images = self._input_images + self.images_blur[i] - tf.stop_gradient(self.images_blur[i]) 
            	
            self._input_data = []

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[self.batchsize, 4], name="input_control"))

            self.labels_init()
            
            self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):
            self._network_tensor, self.gradients_steers, self.gradients_acc, self.gradients_brake = load_imitation_learning_network(self._input_images,
                                                                   self._input_data,
                                                                   self._image_size, self._dout)

       	self.predictions = []
        for i in range(self.batchsize):
        	direction_tensor = self.directions[i][0]
        	prediction_this = tf.case(pred_fn_pairs=[
        		(tf.equal(direction_tensor,2), lambda:self._network_tensor[0][i][0]),
        		(tf.equal(direction_tensor,0.0), lambda:self._network_tensor[0][i][0]),
        		(tf.equal(direction_tensor,3), lambda:self._network_tensor[2][i][0]),
        		(tf.equal(direction_tensor,4), lambda:self._network_tensor[3][i][0])],
        		default=lambda:self._network_tensor[1][i][0],
        		exclusive=False)
        	self.predictions.append(tf_rad2deg(prediction_this))

       	self.predictions = tf.convert_to_tensor(self.predictions)
       	self.predictions = tf.reshape(self.predictions,[self.batchsize,1])

        self.define_loss()
        
        self.optimizer = tf.train.AdamOptimizer(self.LR,beta1=0)
       	
       	self.grads_and_vars = self.optimizer.compute_gradients(self.loss,[self.adversary_parameters])
       	self.grads_image = self.optimizer.compute_gradients(self.loss,self.image_inputs_actual)[0][0]
       	self.rgb_grads_adversaries = []

       	## apply ccm to gradients
       	self.grads_image_CCM = []
       	self.d_adversaries = []
       	c = self.coordinates_placeholder[...,None]*self.grads_image[None,...]
       	c_sum = tf.reduce_sum(c,[2,3])
       	c_sum = tf.expand_dims(c_sum, 2)
       	self.CCMs_transpose_withoutbias_adversaries = tf.stack(c_sum.shape[0]*[self.CCMs_transpose_withoutbias],axis=0)
       	self.d = tf.matmul(c_sum,self.CCMs_transpose_withoutbias_adversaries) 
       	self.valid_grads_adversaries = tf.reduce_sum(self.d,[1,2])
       	self.valid_grads_adversaries = tf.reshape(self.valid_grads_adversaries,[-1])
       	self.grads_and_vars_color = []
       	
       	for i in range(self.valid_grads_adversaries.shape[0]):
       		self.grads_and_vars_color.append(tuple([self.valid_grads_adversaries[i],self.color_parameters[i]]))
       	
       	self.opt_apply_color = self.optimizer.apply_gradients(self.grads_and_vars_color)
       	self.opt_apply = self.optimizer.apply_gradients(self.grads_and_vars)

        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'

        self._sess.run(tf.global_variables_initializer())

        self.load_model()

        self._image_cut = image_cut

        self.weights_parameters_num = self._sess.run(self.weights_parameters)

        self.parameters_actual_tmp, self.parameters_blur_tmp, self.colors_actual_num = self._sess.run((self.adversary_parameters, self.adversary_parameters_blur,self.color_parameters))
        self.parameters_actual_num = [self.parameters_actual_tmp[i]*self.norm_factors[i] for i in range(len(self.parameters_actual_tmp))]
        self.parameters_blur_num = []
        for i in range(len(self.parameters_blur_tmp)):
        	adversary_parameter = self.parameters_blur_tmp[i]
        	norm_factor = self.norm_factors[i]
        	a = [int(round(adversary_parameter[j]*norm_factor)) for j in range(len(adversary_parameter))]
        	if a[0] == norm_factor:
        		a[1] = a[0]
        		a[0] = a[0] - self.step_factors[i]
        	self.parameters_blur_num.append(a)
        
        self.adversary_control_function()
    	
    def get_hm_cleanFrames(self):
    	self.framesInfos = []
    	self.clean_frames = []
    	self.CCMs = []
    	self.CCMs_transpose_withoutbias = []
    	self.trajectory_noFrames = []
    	self.h_my_infos = []

    	trajectory_noFrame_before = 0
    	
    	for trajectory_no in self.trajectories:
    		if trajectory_no == 1:
    			# clean_trajectory 
    			dataset_pt = self.dataset_rt_clean + "datasets_1/"+self.task+self.intersection
    			ccm_pt = dataset_pt + "CCM/"
    		else:
    			dataset_pt = self.dataset_rt_noises + "datasets_"+str(trajectory_no)+"/"+self.task+self.intersection
    		
    		input_info_pt = dataset_pt + "inputInfo/"
    		framesInfo_pt = dataset_pt + "framesInfo/"
    		clean_frames_pt = dataset_pt + "clean_frame/"

    		frames_num = os.listdir(framesInfo_pt)
    		frames_num.sort(key=lambda fname: int(fname.split('.')[0]))
    		frame_range = [int(x.split(".")[0]) for x in frames_num]
    		trajectory_noFrame_before += len(frame_range)
    		self.trajectory_noFrames.append(trajectory_noFrame_before)
    		for i in range(len(frame_range)):
    			frame_num = frame_range[i]
    			CCM_npzfile = np.load(ccm_pt+"ccm.npz")
    			self.CCMs.append(CCM_npzfile["CCM"])
    			self.CCMs_transpose_withoutbias.append(CCM_npzfile["CCM"][:3,:].T)
    			
    			frameInfo_file = framesInfo_pt + str(frame_num) + ".pickle"
    			with open(frameInfo_file,"rb") as handle:
    				frameInfo_this = pickle.load(handle)
    				h_my = np.array(frameInfo_this["h_cv2"])
    			h_my_info = prepWarp_crop_resize(h_my)
    			self.framesInfos.append(h_my)
    			self.h_my_infos.append(h_my_info)
    			
    			clean_frame_file = clean_frames_pt + str(frame_num) + ".png"
    			clean_frame = np.float32(imread(clean_frame_file))/255.
    			self.clean_frames.append(clean_frame)
    		
    	self.framesInfos = np.array(self.framesInfos)
    	self.h_my_infos = np.array(self.h_my_infos)
    	self.clean_frames = np.array(self.clean_frames)
    	self.CCMs = np.float32(self.CCMs)
    	self.CCMs_transpose_withoutbias = np.float32(self.CCMs_transpose_withoutbias)
    	self.batchsize = len(self.framesInfos)
    
    def labels_init(self):
    	directions_input = []
    	speeds = []
    	labels_input = []
    	for trajectory_no in self.trajectories:
    		if trajectory_no == 1:
    			# clean_trajectory 
    			dataset_pt = self.dataset_rt_clean + "datasets_1/"+self.task+self.intersection
    			steer_pt = dataset_pt + "steer/"
    		else:
    			dataset_pt = self.dataset_rt_noises + "datasets_"+str(trajectory_no)+"/"+self.task+self.intersection
    			steer_pt = dataset_pt + "steer_deviation_cleanFrame/"
    		
    		input_info_pt = dataset_pt + "inputInfo/"
    		framesInfo_pt = dataset_pt + "framesInfo/"

    		frames_num = os.listdir(framesInfo_pt)
    		frames_num.sort(key=lambda fname: int(fname.split('.')[0]))
    		frame_range = [int(x.split(".")[0]) for x in frames_num]
    		for i in range(len(frame_range)):
    			frame_count = frame_range[i]
    			input_info_file = input_info_pt + str(frame_count)+".pickle"
    			with open(input_info_file,"rb") as handle:
    				input_info_dict = pickle.load(handle)
    			speed = input_info_dict["forward_speed"]
    			speed = np.array(speed / 25.0)
    			speeds.append(speed)
    			direction = input_info_dict["directions"]
    			directions_input.append(direction)
    			steer_file = steer_pt + str(frame_count)+".pickle"
    			with open(steer_file,"rb") as handle:
    				steer_dict = pickle.load(handle)
    				steer_true = tf_rad2deg(steer_dict["steer"])
    			labels_input.append(steer_true)
    	
    	directions_input = np.array(directions_input)
    	speeds = np.array(speeds)
    	labels_input = np.array(labels_input)
    	self.input_speed = tf.constant(speeds,shape=(self.batchsize,1),dtype=tf.float32,name="input_speed")
    	self._input_data.append(self.input_speed)
    	self.directions = tf.constant(directions_input,shape=(self.batchsize,1),dtype=tf.float32,name="labels")
    	self.labels = tf.constant(labels_input,shape=(self.batchsize,1),dtype=tf.float32,name="labels")

    def define_loss(self):
    	self.predictions1 = self.predictions[:self.trajectory_noFrames[0]]
    	self.predictions2 = self.predictions[self.trajectory_noFrames[0]:self.trajectory_noFrames[1]]
    	self.predictions3 = self.predictions[self.trajectory_noFrames[1]:self.trajectory_noFrames[2]]
    	self.predictions4 = self.predictions[self.trajectory_noFrames[2]:self.trajectory_noFrames[3]]
    	
    	self.labels1 = self.labels[:self.trajectory_noFrames[0]]
    	self.labels2 = self.labels[self.trajectory_noFrames[0]:self.trajectory_noFrames[1]]
    	self.labels3 = self.labels[self.trajectory_noFrames[1]:self.trajectory_noFrames[2]]
    	self.labels4 = self.labels[self.trajectory_noFrames[2]:self.trajectory_noFrames[3]]

    	self.loss_sum1 = tf.reduce_sum(tf.subtract(self.predictions1,self.labels1))
    	self.loss_sum1 = tf.reshape(self.loss_sum1,[-1])
    	self.loss1 = -tf.abs(self.loss_sum1)
    	
    	self.loss_sum2 = tf.reduce_sum(tf.subtract(self.predictions2,self.labels2))
    	self.loss_sum2 = tf.reshape(self.loss_sum2,[-1])
    	self.loss2 = -tf.abs(self.loss_sum2)
    	
    	self.loss_sum3 = tf.reduce_sum(tf.subtract(self.predictions3,self.labels3))
    	self.loss_sum3 = tf.reshape(self.loss_sum3,[-1])
    	self.loss3 = -tf.abs(self.loss_sum3)
    	
    	self.loss_sum4 = tf.reduce_sum(tf.subtract(self.predictions4,self.labels4))
    	self.loss_sum4 = tf.reshape(self.loss_sum4,[-1])
    	self.loss4 = -tf.abs(self.loss_sum4)
    	
    	self.loss = self.loss1 + self.loss2 + self.loss3 + self.loss4
    
    def load_model(self):
        variables_to_restore = tf.global_variables()
        del_index = []
        for i in range(len(variables_to_restore)):
        	if variables_to_restore[i].op.name.startswith("adversary"):
        		del_index.append(i)
        	if variables_to_restore[i].op.name.startswith("color"):
        		del_index.append(i)
        	elif variables_to_restore[i].op.name.startswith("beta"):
        		del_index.append(i)

        variables_to_restore = np.array(variables_to_restore)
        variables_to_restore = np.delete(variables_to_restore, del_index,0)
        variables_to_restore = list(variables_to_restore)
      
        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0
        return ckpt
    	
    def get_adversary(self,name,value,CCM,colors_name,colors_value):

    	numbers = {}
    	for adversary_name in name:
    		numbers[adversary_name] = {}
    		numbers[adversary_name]['rot'] = 0
    		numbers[adversary_name]['pos'] = 88
    		numbers[adversary_name]['width'] = 10
    		numbers[adversary_name]['length'] = 40
    		numbers[adversary_name]['r'] = 0
    		numbers[adversary_name]['g'] = 0
    		numbers[adversary_name]['b'] = 0
    	
    	count = 0
    	for adversary_name in name:
    		name_this = name[adversary_name]
    		for i in range(len(name_this)):
    			numbers[adversary_name][name_this[i]] = value[count]
    			count += 1
    	
    	count = 0
    	for adversary_name in colors_name:
    		name_this = colors_name[adversary_name]
    		for i in range(len(name_this)):
    			numbers[adversary_name][name_this[i]] = colors_value[count]
    			count += 1
    	
    	for adversary_name in name:
    		r = numbers[adversary_name]["r"]
    		g = numbers[adversary_name]["g"]
    		b = numbers[adversary_name]["b"]

    		color_this = [r,g,b,1]
    		color_this = np.float32(color_this)
    		color_transform = np.matmul(color_this, CCM)
    		color_transform = np.clip(color_transform,0,1)
    		color_transform = np.int32(color_transform*255.)
    		r = int(color_transform[0])
    		g = int(color_transform[1])
    		b = int(color_transform[2])
    		numbers[adversary_name]["r"] = r 
    		numbers[adversary_name]["g"] = g
    		numbers[adversary_name]["b"] = b
    	
    	adversaries_res = []
    	for adversary_name in numbers:
    		adversaries_res.append(self.draw.draw_line(numbers[adversary_name]))
    	adversaries_res = np.array(adversaries_res)
    	return adversaries_res
    
    def get_adversary_approximate(self,name,value,CCM,colors_name,colors_value):

    	numbers = {}
    	for adversary_name in name:
    		numbers[adversary_name] = {}
    		numbers[adversary_name]['rot'] = 0
    		numbers[adversary_name]['pos'] = 88
    		numbers[adversary_name]['width'] = 10
    		numbers[adversary_name]['length'] = 40
    		numbers[adversary_name]['r'] = 0
    		numbers[adversary_name]['g'] = 0
    		numbers[adversary_name]['b'] = 0
    	
    	count = 0
    	for adversary_name in name:
    		name_this = name[adversary_name]
    		for i in range(len(name_this)):
    			numbers[adversary_name][name_this[i]] = value[count]
    			count += 1
    	
    	count = 0
    	for adversary_name in colors_name:
    		name_this = colors_name[adversary_name]
    		for i in range(len(name_this)):
    			numbers[adversary_name][name_this[i]] = colors_value[count]
    			count += 1
    	
    	for adversary_name in name:
    		r = numbers[adversary_name]["r"]
    		g = numbers[adversary_name]["g"]
    		b = numbers[adversary_name]["b"]

    		color_this = [r,g,b,1]
    		color_this = np.float32(color_this)
    		color_transform = np.matmul(color_this, CCM)
    		color_transform = np.clip(color_transform,0,1)
    		color_transform = np.int32(color_transform*255.)
    		r = int(color_transform[0])
    		g = int(color_transform[1])
    		b = int(color_transform[2])
    		numbers[adversary_name]["r"] = r 
    		numbers[adversary_name]["g"] = g
    		numbers[adversary_name]["b"] = b
    	
    	return self.draw.draw_lines(numbers) 
    
    def get_adversary_frame(self,name,value,frame_idx,colors_name,colors_value):
    	h_my_info = copy.deepcopy(self.h_my_infos[frame_idx])
    	clean_frame = copy.deepcopy(self.clean_frames[frame_idx])
    	im_res = clean_frame
    	CCM = self.CCMs[frame_idx]
    	adversaries_res = self.get_adversary(name,value,CCM,colors_name,colors_value)
    	adversaries_coordinates_perFrame = []
    	for i in range(adversaries_res.shape[0]):
    		adversary_image = adversaries_res[i]
    		adversary_image = np.float32(adversary_image)/255.
    		im_out = doWarp(adversary_image,h_my_info,(clean_frame.shape[1],clean_frame.shape[0]))
    		alpha  = im_out[:,:,3]
    		im_res = alpha[...,None]*im_out[:,:,:3] + (1.0-alpha[...,None])*im_res[:,:,:3]
    		adversaries_coordinates_perFrame.append(alpha)
    	
    	adversaries_coordinates_perFrame = np.array(adversaries_coordinates_perFrame)
    	
    	masks_per_frame = []
    	for i in range(adversaries_coordinates_perFrame.shape[0]):
    		coordinates_this = adversaries_coordinates_perFrame[i]
    		coordinates_forward_rest = np.sum(adversaries_coordinates_perFrame[(i+1):],axis=0)
    		coordinates_no_overlap = coordinates_this-coordinates_forward_rest
    		indices = coordinates_no_overlap > 0

    		masks = np.zeros((self._image_size[0],self._image_size[1]))
    		masks[indices] = 1
    		masks_per_frame.append(masks)
    	masks_per_frame = np.array(masks_per_frame)
    	return im_res,masks_per_frame 
    
    def get_adversary_frame_approximate(self,name,value,frame_idx,colors_name,colors_value):
    	h_my_info = copy.deepcopy(self.h_my_infos[frame_idx])
    	clean_frame = copy.deepcopy(self.clean_frames[frame_idx])
    	im_res = clean_frame
    	CCM = self.CCMs[frame_idx]
    	adversary_image = self.get_adversary_approximate(name,value,CCM,colors_name,colors_value)
    	adversary_image = np.float32(adversary_image)/255.
    	im_out = doWarp(adversary_image,h_my_info,(clean_frame.shape[1],clean_frame.shape[0]))
    	alpha  = im_out[:,:,3]
    	im_res = alpha[...,None]*im_out[:,:,:3] + (1.0-alpha[...,None])*im_res[:,:,:3]
    	return im_res

    def getFrames(self, name, value, colors_name, colors_value):
    	frames_list = []
    	coordinates_frames = []
    	for i in range(self.batchsize):
    		image_warped_this, coordinates = self.get_adversary_frame(name,value,i,colors_name,colors_value)
    		coordinates_frames.append(coordinates)
    		image_warped_this = image_warped_this.astype(np.float32)
    		frames_list.append(image_warped_this)
    	coordinates_frames = np.array(coordinates_frames)
    	coordinates_frames = np.moveaxis(coordinates_frames, 0, 1)
    	return frames_list, coordinates_frames
    
    def getFrames_approximate(self, name, value, colors_name, colors_value):
    	frames_list = []
    	coordinates_frames = []
    	for i in range(self.batchsize):
    		image_warped_this = self.get_adversary_frame_approximate(name,value,i,colors_name,colors_value)
    		image_warped_this = image_warped_this.astype(np.float32)
    		frames_list.append(image_warped_this)
    	return frames_list 
    
    def adversary_control_function(self,epoch=1000):
    	input_speed = self._input_data[1]
    	dout = self._dout
    	labels = self.labels
    	branches = self._network_tensor
    	while self.iter <= self.epoch:
    		speeds = []
    		labels_input = []
    		directions_input = []
    		steers_predictions = []

    		frames_actual,coordinates_frames = self.getFrames(self.name,self.parameters_actual_num,self.colors_name,self.colors_actual_num)

    		frames_parameters_floor = {}
    		for i in range(len(self.parameters_actual_num)):
    			parameters_this = np.array(copy.deepcopy(self.parameters_actual_num))
    			parameters_this[i]= self.parameters_blur_num[i][0]
    			frames_parameters_floor[i] = self.getFrames_approximate(self.name,parameters_this,self.colors_name,self.colors_actual_num)
    			
    		# ceil frames
    		frames_parameters_ceil = {}
    		for i in range(len(self.parameters_actual_num)):
    			parameters_this = np.array(copy.deepcopy(self.parameters_actual_num))
    			parameters_this[i] = self.parameters_blur_num[i][1]
    			frames_parameters_ceil[i] = self.getFrames_approximate(self.name,parameters_this,self.colors_name,self.colors_actual_num)
    		
    		# get the learning rate
    		feed_dict = {}
    		
    		feed_dict[dout] = [1]*len(self.dropout_vec)

    		feed_dict[self.coordinates_placeholder] = coordinates_frames
    		
    		# feed images
    		feed_dict[self.image_inputs_actual] = frames_actual
    		#feed_dict[self.image_inputs_actual] = self.clean_frames
    		for i in range(len(self.parameters_actual_num)):
    			feed_dict[self.image_inputs_floor[i]] = frames_parameters_floor[i]
    			feed_dict[self.image_inputs_ceil[i]] = frames_parameters_ceil[i]
    		
    		loss_num,predictions_num,_,_ = self._sess.run((self.loss,self.predictions,self.opt_apply,self.opt_apply_color),feed_dict=feed_dict)
    		
    		predictions_num_print = predictions_num
    		predictions_num_print = np.array(predictions_num_print)
    		print("start_prediction")
    		predictions_num_print = np.reshape(predictions_num_print,(len(predictions_num_print),1))
    		print(predictions_num_print)
    		print("end_prediction")

    		count = 0
    		count1 = 0
    		for adversary_name in self.name:
    			for i in range(len(self.name[adversary_name])):
    				sys.stdout.write("%s=%.2f,"%(self.name[adversary_name][i],self.parameters_actual_num[count]))
    				count += 1
    			for j in range(len(self.colors_name[adversary_name])):
    				sys.stdout.write("%s=%.2f,"%(self.colors_name[adversary_name][j],self.colors_actual_num[count1]*255.))
    				count1 += 1
    		sys.stdout.write("LR=%.5f,"%(self.LR))
    		sys.stdout.write("loss=%.3f"%(loss_num))
    		sys.stdout.write("\n")
    		
    		count = 0
    		tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
    		sys.stdout.write(tmstr + ",iter=[%09d] \n" % (self.iter))
    		
    		sys.stdout.write("\n\n")
    		sys.stdout.flush()
    		
    		self.parameters_actual_tmp,self.colors_actual_num,self.parameters_blur_normed = self._sess.run((self.adversary_parameters,self.color_parameters,self.adversary_parameters_blur))
    		self.parameters_actual_num = [self.parameters_actual_tmp[i]*self.norm_factors[i] for i in range(len(self.parameters_actual_tmp))]
    		
    		self.parameters_blur_num = []
    		for i in range(len(self.parameters_blur_normed)):
    			adversary_parameter = self.parameters_blur_normed[i]
    			norm_factor = self.norm_factors[i]
    			a = [int(round(adversary_parameter[j]*norm_factor)) for j in range(len(adversary_parameter))]
    			if a[0] == norm_factor:
    				a[1] = a[0]
    				a[0] = a[0] - self.step_factors[i]
    			self.parameters_blur_num.append(a)
    			
    		self.iter = self.iter + 1

    	tf.reset_default_graph()
    
