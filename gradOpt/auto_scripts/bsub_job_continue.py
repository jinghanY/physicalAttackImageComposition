import os
import shutil
import numpy as np
import glob
import random
import argparse
# parameters to be changed 

def select_random(inputList):
	return inputList[int(random.uniform(0,len(inputList)))]

argparser = argparse.ArgumentParser()
argparser.add_argument("-a","--intersection",type=str)
argparser.add_argument("-t","--task",type=str)
argparser.add_argument("-n","--no_para",type=int)
argparser.add_argument("-s","--stepSize1",type=int)
argparser.add_argument("-f","--stepSize2",type=int)
argparser.add_argument("-g","--stepSize3",type=int)
args = argparser.parse_args()

LR = 0.1
step = 100
randomSeed = 1801
intersection = args.intersection+"/"
task = args.task+"/"
step_size1 = int(args.stepSize1)
step_size2 = int(args.stepSize2)
step_size3 = int(args.stepSize3)
no_paras = [args.no_para]
no_paras_num = no_paras[0]
## write config
config_pt = "config_"+str(no_paras_num)+"/"
intersection_path = config_pt + task + intersection

outfile_pt = "out_files/"
outfile_task_intersection_pt = outfile_pt + task + intersection

if not os.path.exists(intersection_path):
	os.makedirs(intersection_path)

if not os.path.exists(outfile_task_intersection_pt):
	os.makedirs(outfile_task_intersection_pt)

def write_config(intersection_path, task, intersection, adversary,color,num,LR,step,randomSeed,step_size1,step_size2,step_size3):
	fname= intersection_path + task.strip("/")+"-"+intersection.strip("/")+"-" +str(num)+ ".json"
	with open(fname,"w") as f:
		f.write("{\n")
		f.write("\"task\":"+"\""+task+"\""+",\n")
		f.write("\"intersection\":"+"\""+intersection+"\""+",\n")
		f.write("\"LR\":"+"\""+str(LR)+"\""+",\n")
		f.write("\"randomSeed\":"+"\""+str(randomSeed)+"\""+",\n")
		f.write("\"step\":"+"\""+str(step)+"\""+",\n")
		f.write("\"stepSize1\":"+"\""+str(step_size1)+"\""+",\n")
		f.write("\"stepSize2\":"+"\""+str(step_size2)+"\""+",\n")
		f.write("\"stepSize3\":"+"\""+str(step_size3)+"\""+",\n")
		f.write("\"adversary\":{\n")
		
		count = 1
		for adversary_name in adversary:
			f.write("\""+adversary_name+"\":")
			f.write("[")
			count2 = 1
			for param in adversary[adversary_name]:
				f.write("\""+param+"\"")
				if count2 != len(adversary[adversary_name]):	
					f.write(",")
					count2 += 1
			if count == len(color.keys()):
				f.write("]")
			else:	
				f.write("],")
			
			count += 1
			f.write("\n")
			
		f.write("},\n")
		f.write("\"color\":{\n")
		
		count = 1
		for adversary_name in color:
			f.write("\""+adversary_name+"\":")
			f.write("[")
			count2 = 1
			for param in color[adversary_name]:
				f.write("\""+param+"\"")
				if count2 != len(color[adversary_name]):	
					f.write(",")
					count2 += 1
			
			if count == len(color.keys()):
				f.write("]")
			else:	
				f.write("],")
			
			count += 1
			f.write("\n")
		
		f.write("}\n")
		f.write("}\n")

names = ["rectangle1","rectangle2","rectangle3","rectangle4","rectangle5","rectangle6","rectangle7","rectangle8","rectangle9","rectangle10","rectangle11","rectangle12","rectangle13","rectangle14","rectangle15","rectangle16","rectangle17","rectangle18","rectangle19","rectangle20","rectangle21","rectangle22","rectangle23","rectangle24"]
paras = ["rot","pos","width","length"]
color_paras = ["r","g","b"]

count = 1
count2 = 0
adversaries = {}
colors = {}
no_3 = np.array(no_paras)/7
no_3 = int(no_3[0])
no_paras_array_config = list(np.int32(4*np.array(no_paras)/7))
print(no_paras_array_config)
for adversary_name in names:
	list_this = []
	color_list_this = []
	for j in range(len(paras)):
		list_this.append(paras[j])
		try:
			color_list_this.append(color_paras[j])
		except:
			pass
		adversaries[adversary_name] = list_this 
		colors[adversary_name] = color_list_this
		if count in no_paras_array_config: 
			write_config(intersection_path, task, intersection, adversaries, colors, count+3*no_3, LR, step, randomSeed,step_size1,step_size2,step_size3)
			count2 += 1
		count += 1

## write bsub jobs
def write_gpuJob(json_file, outfile_pt, gpuJob_pt, script_name,LR):
	json_info = json_file.split("/")
	task = json_info[1]
	intersection = json_info[2]
	no_paras_str = json_info[0].split("_")[-1]
	outfile_task_intersection_pt = outfile_pt + task + "/" + intersection + "/"
	
	#no_paras_str = json_file.strip(".json").split("-")[-1]
	outfile = outfile_task_intersection_pt + no_paras_str + ".txt"

	gpuJob_task_intersection_pt = gpuJob_pt + task + "/" + intersection + "/" 
	gpuJob = gpuJob_task_intersection_pt + no_paras_str
	if not os.path.exists(gpuJob_task_intersection_pt):
		 os.makedirs(gpuJob_task_intersection_pt)
	
	#print(outfile)

	## we don't need the input of gpuJob, gpuJob_task_intersection_pt
	#print(script_name)
	#print(json_info)
	#print(json_file)

	with open(gpuJob,"w") as f:	
		#f.write("#BSUB -G SEAS-Lab-Vorobeychik\n")
		#f.write("#BSUB -o "+gpuJob_task_intersection_pt+"mycpucode_out.%J\n")
		#f.write("#BSUB -N\n")
		#f.write("#BSUB -J PythonCPUJob\n")
		#f.write("#BSUB -n 2\n")
		#f.write("#BSUB -R \'(!gpu) span[hosts=1] affinity[core(1)]\'\n")
		#f.write("taskset -c $LSB_BIND_CPU_LIST bash -c \'python3 " + script_name + " -c "+json_file+" >> "+outfile+"\'")
		f.write("python3 " + script_name + " -c "+json_file+" >> "+outfile)

	return gpuJob

def bash_subJob(job_ids):
	fname = "sbj"+str(no_paras_num)+".sh"
	with open(fname,"w") as f:
		f.write("#!/bin/bash\n")
		for i in range(len(job_ids)):
			job_id = job_ids[i]
			f.write("bsub < "+job_id+"\n")

json_files = []

json_path = config_pt + task + intersection
for path, subdirs, files in os.walk(config_pt):
    for name in files:
            json_files.append(os.path.join(path, name))

#json_files = glob.glob(json_path+"*.json")
json_files.sort(key=lambda fname: int(fname.split('.')[0].split("-")[-1]))


if not os.path.exists(outfile_task_intersection_pt):
	os.makedirs(outfile_task_intersection_pt)

script_name = "imitation_fly_auto_continue/run.py" 

gpuJob_pt = "Jobs_"+str(no_paras_num)+"/"
if not os.path.exists(gpuJob_pt):
	os.makedirs(gpuJob_pt)

gpuJob_task_intersection_pt = gpuJob_pt + task + intersection
if not os.path.exists(gpuJob_task_intersection_pt):
	os.makedirs(gpuJob_task_intersection_pt)

job_ids = []

for i in range(len(json_files)):	
	json_file = json_files[i]
	job_id = write_gpuJob(json_file, outfile_pt, gpuJob_pt, script_name,LR)
	job_ids.append(job_id)

#bash_subJob(job_ids)
