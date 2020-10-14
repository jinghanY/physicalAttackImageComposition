import numpy as np
import re
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
