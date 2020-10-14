import os
import argparse
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--task','-t',help='task')
parser.add_argument('--intersection','-a',help='intersection')
args =parser.parse_args()
task = str(args.task)
intersection = str(args.intersection)
out_pt= "out_files/"+task+"/"+intersection+"/"

def create(dir_pt):
	if not os.path.exists(dir_pt):
		os.makedirs(dir_pt)

create(out_pt)
