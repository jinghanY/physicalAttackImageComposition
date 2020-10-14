import numpy as np
import os
import argparse
import shutil

def readFile(fileName):
	f = open(fileName,"r")
	lines = f.readlines()
	parameters = []
	losses = []
	for line in lines:
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
	return parameters, losses

paras_names = ["rot","pos","width","length","r","g","b"]

# write the json file for checking 
def writeJson(fileName,paras):
	f = open(fileName,"w")
	f.write("{\n")
	for i in range(len(paras)):
		adversary_name = "rectangle"+str(i+1)
		f.write("\""+adversary_name+"\"" + ":{")
		paras_this = paras[i]
		for j in range(len(paras_this)):
			para_name = paras_names[j]
			para_value = paras_this[j]
			if j < len(paras_this) - 1:
				f.write("\""+para_name+"\""+":"+str(para_value)+",")
			elif i < len(paras)-1:
				f.write("\""+para_name+"\""+":"+str(para_value)+"},\n")
			else:	
				f.write("\""+para_name+"\""+":"+str(para_value)+"}\n")
	f.write("}\n")

json_pt = "jsonFiles/"

#argparser = argparse.ArgumentParser()
#argparser.add_argument("-n","--no_para",type=int)
#args = argparser.parse_args()

#no_para = args.no_para
no_para = 7
json_pt = json_pt + str(no_para)+"/"
if os.path.exists(json_pt):
	shutil.rmtree(json_pt)
os.makedirs(json_pt)

outFile_pt = "out_files_final_version/"+"out_files_" + str(no_para)
outFiles = []

for path, subdirs, files in os.walk(outFile_pt):
	for name in files:
		outFiles.append(os.path.join(path,name))
for outFile in outFiles:
	elements = outFile.split("/")
	task = elements[2]
	intersection = elements[3]
	json_file = json_pt + intersection+"-" + task+ "-" + str(no_para) + ".json"
	paras_this, losses_this = readFile(outFile)
	paras_this = np.array(paras_this)
	losses_this = np.array(losses_this)
	idx = np.argsort(losses_this)[0]
	paras_this = paras_this[idx]
	loss_this = losses_this[idx]
	para_this = paras_this
	#para_this = paras_this[-1]
	writeJson(json_file,para_this)
