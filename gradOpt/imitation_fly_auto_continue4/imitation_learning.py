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


def parameter_one_randomGen(x,y):
	"""
	Generate a random number between x and y
	precision 0.1 
	"""
	return math.floor(random.uniform(x, y) * 10) / 10

def para_gen(para_name,step_size):
	if para_name.startswith("rot"):
		return [parameter_one_randomGen(20,180),180.0,step_size]
	elif para_name.startswith("pos"):
		return [parameter_one_randomGen(0,380),380.0,step_size]
	elif para_name.startswith("width"): 
		return [parameter_one_randomGen(0,100),100.0,step_size]
	elif para_name.startswith("length"):
		return [parameter_one_randomGen(0,70),70.0,step_size]
	elif para_name.startswith("r"):
		return [parameter_one_randomGen(0,255),255.0,step_size]
	elif para_name.startswith("g"):
		return [parameter_one_randomGen(0,255),255.0,step_size]
	elif para_name.startswith("b"):
		return [parameter_one_randomGen(0,255),255.0,step_size]

def dotR(inputStr):
	return str(inputStr).replace(".","-")
    	
def create_init_paras(paras_name,step_size):
	paras_d = {}
	paras_d["name"] = []
	paras_d["value"] = []
	paras_d["norm_factor"] = []
	paras_d["step_factor"] = []

	for para_name in paras_name:
		init_value,norm_factor,step_factor = para_gen(para_name,step_size)
		#init_value = paras_name[para_name]
		paras_d["name"].append(para_name)
		paras_d["value"].append(float(init_value/norm_factor))
		paras_d["norm_factor"].append(float(norm_factor))
		paras_d["step_factor"].append(float(step_factor))
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

class ImitationLearning():

    def __init__(self, paras_dict, epoch, step_size,memory_fraction=0.25, image_cut=[115, 510],LR=0.1,task='right-turn/',intersection="42_47/"):

        self.epoch = epoch
        self.step_size = step_size
        self.paras_ds = {}
        self.task = task
        self.intersection = intersection
        self.trajectories = [1,2,3,4]
        for adversary_name in paras_dict:
        	self.paras_d = create_init_paras(paras_dict[adversary_name],step_size)
        	self.paras_ds[adversary_name] = self.paras_d
        
        # continue the experiments
        para_default = [[0,88,10,40,0,0,0]]
        no_para_training = 0
        for adversary_name in self.paras_ds:
        	no_para_training += len(self.paras_ds[adversary_name]["name"])
        
        self.iter = 0
        self.parameters_previous_unnormed = None
        try:
        	fileName_input = "out_files/"+task+intersection+str(no_para_training)+".txt"
        	parameters_input, losses_input, iteration_number = readFile(fileName_input)
        	parameters_input = np.array(parameters_input)
        	losses_input = np.array(losses_input)
        	idx = np.argsort(losses_input)[0]
        	iteration_number = np.array(iteration_number)
        	paras_input = parameters_input[-1]
        	self.iter = iteration_number[-1] + 1
        	no_adversary = int(np.ceil(no_para_training/7))
        	para_default_input = np.array(no_adversary*para_default)
        	for j in range(no_adversary):
        		paras_input_one_adversary = paras_input[j]
        		for h in range(len(paras_input_one_adversary)):
        			para_default_input[j][h] = paras_input[j][h]

        	dict_input = {}
        	paras_names = ["rot","pos","width","length",'r','g','b']
        	norm_factors_default = [180.0, 380.0, 100.0, 70.0, 255.0, 255.0, 255.0]
        	step_factor_default = [step_size,step_size,step_size,step_size,step_size,step_size,step_size]
        	#step_factor_default = [60., 60., 40., 40., 60., 60., 60.]
        	for i in range(len(paras_input)):
        		paras_this = paras_input[i]
        		dict_input["rectangle"+str(i+1)] = {}
        		for j in range(len(paras_this)):
        			para_name = paras_names[j]
        			para_value = paras_this[j]
        			dict_input["rectangle"+str(i+1)][para_name] = para_value
        
        	for adversary_name in dict_input:
        		dict_this = dict_input[adversary_name]
        		for name in dict_this:
        			value = dict_this[name]
        			try:
        				idx = self.paras_ds[adversary_name]["name"].index(name)
        				self.paras_ds[adversary_name]["value"][idx] = value
        			except:
        				self.paras_ds[adversary_name]["name"].append(name)
        				self.paras_ds[adversary_name]["value"].append(value)
        				idx = paras_names.index(name)
        				norm_factor_this = norm_factors_default[idx]
        				step_factor_this = step_factor_default[idx]
        				self.paras_ds[adversary_name]["norm_factor"].append(norm_factor_this)
        				self.paras_ds[adversary_name]["step_factor"].append(step_factor_this)
        
        	## apply normalization
        	for adversary_name in self.paras_ds:
        		norm_factor_this = self.paras_ds[adversary_name]["norm_factor"]
        		value_this = self.paras_ds[adversary_name]["value"]
        		for j in range(len(value_this)):
        			self.paras_ds[adversary_name]["value"][j] = float(value_this[j]/norm_factor_this[j])
        
        except:
        	pass
        
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
        
        self.dataset_rt_clean = "../../../gradOpt-color_final/"
        self.dataset_rt_noises = "../../../hgcat_pure_random_noise/"

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

        self.weights_parameters = []
        self.adversary_parameters_blur = []

        self.image_inputs_floor = {}
        self.image_inputs_ceil = {}

        with tf.device(tf_device):
            for i in range(len(self.parameters_init)):
            	var = tf.Variable(initial_value=self.parameters_init[i], name="adversary_"+str(i), dtype="float32",constraint=lambda t: tf.clip_by_value(t, 0, 1))
            	self.adversary_parameters.append(var)
            
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
       	self.grads_image = self.optimizer.compute_gradients(self.loss,self._input_images)[0][0]
       	
       	self.opt_apply = self.optimizer.apply_gradients(self.grads_and_vars)

        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'

        self._sess.run(tf.global_variables_initializer())

        self.load_model()

        self._image_cut = image_cut

        self.weights_parameters_num = self._sess.run(self.weights_parameters)

        self.parameters_actual_tmp, self.parameters_blur_tmp = self._sess.run((self.adversary_parameters, self.adversary_parameters_blur))
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

    	trajectory_noFrame_before = 0
    	
    	for trajectory_no in self.trajectories:
    		if trajectory_no == 1:
    			dataset_pt = self.dataset_rt_clean + "datasets/"+self.task+self.intersection
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
    			CCM_npzfile = np.load(ccm_pt+str(frame_num)+".npz")
    			self.CCMs.append(CCM_npzfile["CCM"])
    			self.CCMs_transpose_withoutbias.append(CCM_npzfile["CCM"][:3,:].T)
    			
    			frameInfo_file = framesInfo_pt + str(frame_num) + ".pickle"
    			with open(frameInfo_file,"rb") as handle:
    				frameInfo_this = pickle.load(handle)
    				h_my = np.array(frameInfo_this["h_cv2"])
    				self.framesInfos.append(h_my)
    			
    			clean_frame_file = clean_frames_pt + str(frame_num) + ".png"
    			clean_frame = np.float32(imread(clean_frame_file))/255.
    			self.clean_frames.append(clean_frame)
    		
    	self.framesInfos = np.array(self.framesInfos)
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
    			dataset_pt = self.dataset_rt_clean + "datasets/"+self.task+self.intersection
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

    def get_adversary(self,name,value,CCM):

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
    	
    	res= self.draw.draw_lines(numbers)
    	
    	return res
    
    def get_adversary_frame(self,name,value,frame_idx):
    	h = copy.deepcopy(self.framesInfos[frame_idx])
    	clean_frame = copy.deepcopy(self.clean_frames[frame_idx])
    	CCM = self.CCMs[frame_idx]
    	adversary_image = self.get_adversary(name,value,CCM)
    	adversary_image = np.float32(adversary_image)/255.
    	h_my_info = prepWarp_crop_resize(adversary_image,h,(clean_frame.shape[1],clean_frame.shape[0]))
    	im_out = doWarp(adversary_image,h_my_info,(clean_frame.shape[1],clean_frame.shape[0]))
    	im_res = clean_frame
    	indices = im_out[:,:,3]>0
    	im_res[indices] = im_out[indices,:3]
    	return im_res
    
    def load_model(self):
        variables_to_restore = tf.global_variables()
        del_index = []
        for i in range(len(variables_to_restore)):
        	if variables_to_restore[i].op.name.startswith("adversary"):
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

    def getFrames(self, name, value):
    	frames_list = []
    	for i in range(self.batchsize):
    		image_warped_this = self.get_adversary_frame(name,value,i)
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

    		frames_actual = self.getFrames(self.name,self.parameters_actual_num)

    		frames_parameters_floor = {}
    		for i in range(len(self.parameters_actual_num)):
    			parameters_this = np.array(copy.deepcopy(self.parameters_actual_num))
    			parameters_this[i]= self.parameters_blur_num[i][0]
    			frames_parameters_floor[i] = self.getFrames(self.name,parameters_this)
    			
    		# ceil frames
    		frames_parameters_ceil = {}
    		for i in range(len(self.parameters_actual_num)):
    			parameters_this = np.array(copy.deepcopy(self.parameters_actual_num))
    			parameters_this[i] = self.parameters_blur_num[i][1]
    			frames_parameters_ceil[i] = self.getFrames(self.name,parameters_this)
    		
    		# get the learning rate
    		feed_dict = {}
    		
    		feed_dict[dout] = [1]*len(self.dropout_vec)
    		
    		#print(self.parameters_actual_num)
    		#print(self.parameters_blur_num)
    		# feed images
    		feed_dict[self.image_inputs_actual] = frames_actual
    		for i in range(len(self.parameters_actual_num)):
    			feed_dict[self.image_inputs_floor[i]] = frames_parameters_floor[i]
    			feed_dict[self.image_inputs_ceil[i]] = frames_parameters_ceil[i]
    		
    		loss_num, predictions_num, _ = self._sess.run((self.loss,self.predictions,self.opt_apply),feed_dict=feed_dict) 
    		
    		print("start_prediction")
    		predictions_num_print = predictions_num
    		predictions_num_print = np.array(predictions_num_print)
    		predictions_num_print = np.reshape(predictions_num_print,(len(predictions_num_print),1))
    		print(predictions_num_print)
    		print("end_prediction")

    		count = 0
    		for adversary_name in self.name:
    			for i in range(len(self.name[adversary_name])):
    				sys.stdout.write("%s=%.2f,"%(self.name[adversary_name][i],self.parameters_actual_num[count]))
    				count += 1
    		sys.stdout.write("LR=%.5f,"%(self.LR))
    		sys.stdout.write("loss=%.3f"%(loss_num))
    		sys.stdout.write("\n")
    		
    		count = 0
    		tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
    		sys.stdout.write(tmstr + ",iter=[%09d] \n" % (self.iter))
    		
    		sys.stdout.write("\n\n")
    		sys.stdout.flush()
    		
    		self.parameters_actual_tmp, self.parameters_blur_normed = self._sess.run((self.adversary_parameters, self.adversary_parameters_blur)) 
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
