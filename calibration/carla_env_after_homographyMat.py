import time
import pandas as pd
import numpy as np
import pickle
from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites.dac_2018_before_homographyMat import DAC2018_before_homographyMat	
from imitation_after_homographyMat.imitation_learning import ImitationLearning
from adversary.shape_homographyMat import Shape
import argparse

WEATHER_DICT = {
        'Default':         0,
        'ClearNoon':       1,
        'CloudyNoon':      2,
        'WetNoon':         3,
        'WetCloudyNoon':   4,
        'MidRainyNoon':    5,
        'HardRainNoon':    6,
        'SoftRainNoon':    7,
        'ClearSunset' :    8,
        'CloudySunset' :   9,
        'WetSunset':       10,
        'WetCloudySunset': 11,
        'MidRainSunset' :  12,
        'HardRainSunset':  13,
        'SoftRainSunset':  14
}

TASKS = {
    'go-straight',
    'turn-right',
    'turn-left'
}

CARLA_PATH = '~/projects/carla-packaged/carla-cluster/CarlaUE4.sh'
POSITION_RANGE = (0, 200)
ROTATION_RANGE = (0, 180)
WIDTH_RANGE = (0, 50)
LENGTH_RANGE = (0, 200)
COLOR_TUPLE_RANGE = (0, 255)

class CarlaEnv:
    def __init__(self, min_frames, max_frames, intersection, trajectory_no, radius,town='Town01_nemesisA',
                task='go_straight', weather='Default',
                port=2000, save_images=False, gpu_num=0,
                experiment_name='baseline',opt=1,color_option= False,save_choice=False,adversary_name="adversary_"):
        """
        Adversary environment for Carla Simulator
        """
        print("Starting CARLA gym environment")
        print("Ensure that CARLA is running on port", port)
        self.town = town
        self.task = task
        self.weather = WEATHER_DICT[weather]
        self.port = port
        self.save_images = save_images
        self.gpu_num = gpu_num
        self.experiment_name = experiment_name
        self.color_option = color_option
        self.radius = radius
        self.save_choice = save_choice
        self.adversary_name = adversary_name
        self.trajectory_no = trajectory_no

        self.min_frames = min_frames
        self.max_frames = max_frames
        
        self.intersection = intersection 

        self.opt = opt

        self.agent = None
        self.avoid_stopping = False
        self.iterations = 1

        # defines what kinds of experiments are going to be run
        self.experiment_suite = DAC2018_before_homographyMat(self.town, self.task, self.weather, self.iterations,intersection=self.intersection)

        # load the adversary generator
        if self.adversary_name == "adversary_":
        	self.adversary_other_name = "adversarybeta_"
        else:
        	self.adversary_other_name = "adversary_"
        
        self.adversary = Shape(self.town, adversary_name=self.adversary_name,radius=self.radius)
        self.adversary_other = Shape(self.town, adversary_name=self.adversary_other_name,radius=self.radius)
        self.step()

    def step(self):
        self.experiment_name = 'adversary_{}'.format(self.opt)
        self.adversary.lines_rotate(opt=self.opt,color_option=self.color_option)
        self.adversary_other.lines_rotate(opt=self.opt,color_option="True")
        self.agent = ImitationLearning(self.town, self.task, self.intersection, self.save_choice,self.avoid_stopping,opt=self.opt,min_frames=self.min_frames,max_frames=self.max_frames, trajectory_no=self.trajectory_no)
        run_driving_benchmark(agent=self.agent, experiment_suite=self.experiment_suite, city_name=self.town, log_name=self.experiment_name, port=self.port) 
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port','-r',help="port number")
    parser.add_argument('--townName','-n',help="town name")
    parser.add_argument('--position','-p',help="position for the red point")
    parser.add_argument('--intersection','-s',help='intersection')
    parser.add_argument('--task','-t',help='task')
    parser.add_argument('--color','-c',help='c')
    parser.add_argument('--saveChoice','-v',help='save',default="False")
    parser.add_argument('--minFrame','-w',help='minFrame',default=0)
    parser.add_argument('--maxFrame','-z',help='maxFrame',default=10000)
    parser.add_argument('--adversaryName','-a',help='save',default="False")
    parser.add_argument('--radius','-u',help='radius',default=1)
    parser.add_argument('--trajectory','-j',help='trajectory',default=1)
    
    args =parser.parse_args()
    townName = str(args.townName)
    portNum = int(args.port)
    opt_this = int(args.position)
    intersection = str(args.intersection)
    task = str(args.task)
    color_option = str(args.color)
    min_frames = int(args.minFrame)
    max_frames = int(args.maxFrame)
    trajectory_no = int(args.trajectory)
    
    saveChoice= str(args.saveChoice)
    adversary_name = str(args.adversaryName)
    radius = int(args.radius)
    env = CarlaEnv(min_frames=min_frames, max_frames=max_frames, intersection=intersection, trajectory_no=trajectory_no,radius=radius, town=townName,task=task,port=portNum, opt=opt_this,color_option=color_option,save_choice=saveChoice,adversary_name=adversary_name)
