from imitation_learning_init import ImitationLearningInit
from imitation_learning import ImitationLearning
from imitation_learningThreshold import ImitationLearningThreshold
import json
import argparse
import os
from readFun import *

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"

argparser = argparse.ArgumentParser()
argparser.add_argument("-c","--config",type=str)
args = argparser.parse_args()
json_name = args.config

# get the iteration number 





with open(json_name) as json_file:
	config = json.load(json_file)

task = config["task"]
intersection = config["intersection"]
adversary_dict = config["adversary"]
LR = float(config["LR"])
step =int(config["step"])
random_seed =int(config["randomSeed"])
step_size1 = int(config["stepSize1"])
step_size2 = int(config["stepSize2"])
step_size3 = int(config["stepSize3"])
no_para_training = 0
for adversary_name in adversary_dict:
	no_para_training += len(adversary_dict[adversary_name])

epoch1=int(step)
epoch2=int(2*step)
epoch3=int(2.5*step)

try:
	fileName_input = "out_files/"+task+intersection+str(no_para_training)+".txt"
	_, _, iteration_number = readFile(fileName_input)
	iteration_number = iteration_number[-1]
except:
	iteration_number = 0

LR1 = LR
LR2 = LR/10
LR3 = LR/100
if iteration_number == 0:
	ImitationLearningInit(paras_dict=adversary_dict,epoch=epoch1,random_seed=random_seed,step_size=step_size1,LR=LR,task=task,intersection=intersection)
	ImitationLearningThreshold(paras_dict=adversary_dict,epoch=epoch2,step_size=step_size2,LR=LR2,task=task,intersection=intersection)
	ImitationLearningThreshold(paras_dict=adversary_dict,epoch=epoch3,step_size=step_size3,LR=LR3,task=task,intersection=intersection)
elif iteration_number < epoch1:
	ImitationLearning(paras_dict=adversary_dict,epoch=epoch1,step_size=step_size1,LR=LR1,task=task,intersection=intersection)
	ImitationLearningThreshold(paras_dict=adversary_dict,epoch=epoch2,step_size=step_size2,LR=LR2,task=task,intersection=intersection)
	ImitationLearningThreshold(paras_dict=adversary_dict,epoch=epoch3,step_size=step_size3,LR=LR3,task=task,intersection=intersection)
elif iteration_number == epoch1:
	ImitationLearningThreshold(paras_dict=adversary_dict,epoch=epoch2,step_size=step_size2,LR=LR2,task=task,intersection=intersection)
	ImitationLearningThreshold(paras_dict=adversary_dict,epoch=epoch3,step_size=step_size3,LR=LR3,task=task,intersection=intersection)
elif epoch1 < iteration_number < epoch2:
	ImitationLearning(paras_dict=adversary_dict,epoch=epoch2,step_size=step_size2,LR=LR2,task=task,intersection=intersection)
	ImitationLearningThreshold(paras_dict=adversary_dict,epoch=epoch3,step_size=step_size3,LR=LR3,task=task,intersection=intersection)
elif iteration_number == epoch2:
	ImitationLearningThreshold(paras_dict=adversary_dict,epoch=epoch3,step_size=step_size3,LR=LR3,task=task,intersection=intersection)
else:	
	ImitationLearning(paras_dict=adversary_dict,epoch=epoch3,step_size=step_size3,LR=LR3,task=task,intersection=intersection)
