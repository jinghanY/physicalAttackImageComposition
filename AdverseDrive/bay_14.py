import os
import json
import time
import numpy as np
import pandas as pd
from carla_env import CarlaEnv
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
import sys
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("-c","--config",type=str)
args = argparser.parse_args()
json_name = args.config

with open('config/'+json_name) as json_file:
    args = json.load(json_file)

# CARLA parameters
curr_task  = args['task']
curr_intersection = args['intersection']
curr_port  = args['port']
curr_gpu   = args['GPU']
curr_town  = args['town']
curr_adversary_name = args['adversary_name']

# bayesian parameters
random_points = args['random_points']
search_points = args['search_points']

acquisition_function = args['acquisition_function']

overwrite_experiment = args['overwrite_experiment']

directory_to_save = './_benchmarks_results/{}'.format(curr_town)
#if os.path.exists(directory_to_save):
#    if overwrite_experiment:
#        print("Removing {}".format(directory_to_save))
#        os.system("rm -rf {}".format(directory_to_save))
#    else:
#        print("ERROR: A directory called {} already exists.".format(directory_to_save))
#        print("Please make sure to move the contents as running this program will overwrite the contents of this directory.")
#        exit()

now = time.time()
print("Loading the Imitition Network and performing one simulation run for the baseline path..")
os.system("mkdir -p _benchmarks_results")
env = CarlaEnv(town=curr_town, task=curr_task, port=curr_port, save_images=False, gpu_num=curr_gpu, adversary_name=curr_adversary_name, intersection=curr_intersection)
print("Complete.")

baseSteer     = env.baseline_steer                   # get the steering angles for the baseline run
MAX_LEN       = int(len(env.baseline_steer)*.8)      # set maximum number of frames to 80 percent of baseline scenario
baseSteer     = baseSteer[:MAX_LEN]                  # subset steering angles to maximum number of allowed frames

    
def target(rot1,pos1,width1,length1,r1,g1,b1,rot2,pos2,width2,length2,r2,g2,b2):
    # specify our attack (in this case double black lines) as a dictionary to pass to the CarlaEnv object.
    # TODO: get the num_para
    numbers = {}

    adversary_name = "rectangle1"
    numbers[adversary_name] = {}
    numbers[adversary_name]['rot'] = 0
    numbers[adversary_name]['pos'] = 88
    numbers[adversary_name]['length'] = 40
    numbers[adversary_name]['width'] = 10
    numbers[adversary_name]['r'] = 0
    numbers[adversary_name]['g'] = 0
    numbers[adversary_name]['b'] = 0
    
    numbers[adversary_name]['rot'] = float(rot1)
    numbers[adversary_name]['pos'] = float(pos1)
    numbers[adversary_name]['length'] = float(length1)
    numbers[adversary_name]['width'] = float(width1)
    numbers[adversary_name]['r'] = float(r1)
    numbers[adversary_name]['g'] = float(g1)
    numbers[adversary_name]['b'] = float(b1)
    
    adversary_name = "rectangle2"
    numbers[adversary_name] = {}
    numbers[adversary_name]['rot'] = 0
    numbers[adversary_name]['pos'] = 88
    numbers[adversary_name]['length'] = 40
    numbers[adversary_name]['width'] = 10
    numbers[adversary_name]['r'] = 0
    numbers[adversary_name]['g'] = 0
    numbers[adversary_name]['b'] = 0
    
    numbers[adversary_name]['rot'] = float(rot2)
    numbers[adversary_name]['pos'] = float(pos2)
    numbers[adversary_name]['length'] = float(length2)
    numbers[adversary_name]['width'] = float(width2)
    numbers[adversary_name]['r'] = float(r2)
    numbers[adversary_name]['g'] = float(g2)
    numbers[adversary_name]['b'] = float(b2)
    
    
    metrics = env.step(numbers)
    
    
    # calculate the objective function we are trying to maximize
    attackSteer = metrics['steer'][:MAX_LEN]

    # if attackSteer vector is shorter than baseSteer, extend attackSteer with baseSteer.
    # This takes care of difference in vector lengths without changing the L1 value
    # as extended part of attackSteer will have zero difference with same part of baseSteer
    if len(attackSteer) < len(baseSteer):
        attackSteer = np.append(attackSteer, baseSteer[len(attackSteer):])
    # return objective function value for this particular run
    diff = np.abs(np.sum(attackSteer - baseSteer))
    sys.stdout.write("rot1=%.2f,pos1=%.2f,width1=%.2f,length1=%.2f,r1=%.2f,g1=%.2f,b1=%.2f,"%(rot1,pos1,width1,length1,r1,g1,b1))
    sys.stdout.write("rot2=%.2f,pos2=%.2f,width2=%.2f,length2=%.2f,r2=%.2f,g2=%.2f,b2=%.2f,"%(rot2,pos2,width2,length2,r2,g2,b2))
    sys.stdout.write("target=%.2f\n"%(diff))
    sys.stdout.flush()
    return diff

controls = {'rot1': (0, 180), 
			'pos1': (0, 400),
            'width1': (0, 100),
            'length1': (0, 70),
            'r1': (0,250),
            'g1': (0,250),
            'b1': (0,250),
            'rot2': (0, 180),
			'pos2': (0, 400),
            'width2': (0, 100),
            'length2': (0, 70),
            'r2': (0,250),
            'g2': (0,250),
            'b2': (0,250)
            }

print("Running the Bayesian Optimizer for {} iterations.".format(str(random_points + search_points)))
# instantiate the bayesian optimizer
optimizer = BayesianOptimization(target, controls, random_state=42)
optimizer.maximize(init_points=random_points, n_iter=search_points, acq=acquisition_function)
print(optimizer.max)
