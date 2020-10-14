import os
import argparse
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--intersection','-i',help='intersection')
parser.add_argument('--task','-t',help='task')
parser.add_argument('--trajectory','-j',help='trajectory')
args =parser.parse_args()
task = str(args.task)
intersection = str(args.intersection)
trajectory_no = int(args.trajectory)
dataset_pt = "datasets_"+str(trajectory_no)+"/"+task+"/"+intersection+"/"
out_pt = "out_done"

if os.path.exists(dataset_pt):
	shutil.rmtree(dataset_pt)

def create(dir_pt):
	if not os.path.exists(dir_pt):
		os.makedirs(dir_pt)

create(dataset_pt)

p1 = dataset_pt + "adversary/"
p2 = dataset_pt + "clean_frame/"
p3 = dataset_pt + "framesInfo/"
p4 = dataset_pt + "inputInfo/"
p5 = dataset_pt + "quality_check/"
p6 = dataset_pt + "steer/"
p7 = dataset_pt + "control_input/"
p8 = dataset_pt + "simulator_frame/"
p9 = out_pt

p = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

for p_this in p:
	create(p_this)
