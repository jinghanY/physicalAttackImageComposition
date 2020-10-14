from __future__ import print_function

import os

import scipy

import tensorflow as tf
import numpy as np

from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import cv2
import time

slim = tf.contrib.slim

from carla.agent import Agent
from carla.carla_server_pb2 import Control
from imitation_homographyMat.imitation_learning_network import load_imitation_learning_network
import random
import pickle
import shutil

"""custom libraries"""

class ImitationLearning(Agent):

    def __init__(self, city_name, task, intersection, save_choice,avoid_stopping, min_frames, max_frames, memory_fraction=0.25, image_cut=[115, 510],opt=0):

        Agent.__init__(self)

        self.opt = opt
        self.save_choice = save_choice
        self.intersection = intersection
        task = task
        self.dataset_pt = "datasets_1/"+task+"/"+self.intersection+"/"
        self.input_pt = self.dataset_pt + "inputInfo/"
        self.input_pt_input = self.dataset_pt + "inputInfo_input/"
        self.images_save_path = "color_"+self.intersection+"/"
        self.steer_pt = self.dataset_pt + "steer/"
        self.steer_pt_input = self.dataset_pt + "steer_input/"
        self.control_info_pt = self.dataset_pt + "control_input/"
        self.frame_count = 0
        self.min_frames = min_frames
        self.max_frames = max_frames

        if not os.path.exists(self.dataset_pt):
        	os.mkdir(self.dataset_pt)
        if not os.path.exists(self.input_pt):
        	os.mkdir(self.input_pt)
        if not os.path.exists(self.steer_pt):
        	os.mkdir(self.steer_pt)
        	
        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

        config_gpu = tf.ConfigProto()

        # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES

        config_gpu.gpu_options.visible_device_list = '0'

        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction

        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping

        """ loading adversary """
        self.city_name = city_name
        self._sess = tf.Session(config=config_gpu)

        with tf.device('/gpu:0'):
            self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                                self._image_size[1],
                                                                self._image_size[2]],
                                                name="input_image")

            self._input_data = []

            self.start = True
            self.frame_count = 0


            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 4], name="input_control"))

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 1], name="input_speed"))

            self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):
            self._network_tensor, self.gradients_steers, self.gradients_acc, self.gradients_brake = load_imitation_learning_network(self._input_images,
                                                                   self._input_data,
                                                                   self._image_size, self._dout)

        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'

        # tf.reset_default_graph()
        self._sess.run(tf.global_variables_initializer())

        self.load_model()

        self._image_cut = image_cut

    def load_model(self):

        variables_to_restore = tf.global_variables()

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

    def run_step(self, measurements, sensor_data, directions, target, _episode_number):
        
        self.frame_count += 1
        
        control = self._compute_action(sensor_data['CameraRGB'].data,
                                       measurements.player_measurements.forward_speed,
                                       directions)
        
        return control

    def _compute_action(self, rgb_image, speed, direction=None):

        self.episode_path = "color_"+self.intersection+"/"+"point"+str(self.opt)+"/"
        try:
        	os.makedirs(self.episode_path,exist_ok=True)
        except:
        	pass
        if self.min_frames <= self.frame_count <= self.max_frames:
        	scipy.misc.imsave(self.episode_path+str(self.frame_count)+'.png', rgb_image)

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0], self._image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._control_function(image_input, speed, direction, self._sess)

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        control_info_file = self.control_info_pt + str(self.frame_count)+".pickle"
        control = Control()
        with open(control_info_file,"rb") as handle:
        	control_dict = pickle.load(handle)
        control.steer = control_dict["steer"]
        control.throttle = control_dict["throttle"]
        control.brake = control_dict["brake"]

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input, speed, control_input, sess):

        branches = self._network_tensor
        x = self._input_images
        dout = self._dout
        input_speed = self._input_data[1]

        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))

        # Normalize with the maximum speed from the training set ( 90 km/h)
        speed = np.array(speed / 25.0)

        speed = speed.reshape((1, 1))

        if control_input == 2 or control_input == 0.0:
            all_net = branches[0]
        elif control_input == 3:
            all_net = branches[2]
        elif control_input == 4:
            all_net = branches[3]
        else:
            all_net = branches[1]

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(self.dropout_vec)}

        res_all = sess.run((all_net,branches), feed_dict=feedDict)
        output_all = res_all[0]
        branches_num = res_all[1]
        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        raw_image = np.reshape(image_input, (88, 200, 3))
        

        if self._avoid_stopping:
            predicted_speed = sess.run(branches[4], feed_dict=feedDict)
            predicted_speed = predicted_speed[0][0]
            real_speed = speed * 25.0

            real_predicted = predicted_speed * 25.0
            if real_speed < 2.0 and real_predicted > 3.0:

                predicted_acc = 1 * (5.6 / 25.0 - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc[0][0]

        return predicted_steers, predicted_acc, predicted_brake
