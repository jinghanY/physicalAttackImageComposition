#! /bin/bash
task=${1?Error: no task}
intersection=${2?Error: no intersection}
stepSize1=${3?Error: no stepSize}
stepSize2=${4?Error: no stepSize}
stepSize3=${5?Error: no stepSize}
for i in 7 14 21 28 35
do
	echo $i
	python create_output.py -t $task -a $intersection
	python auto_scripts/bsub_job_continue.py -a $intersection -t $task -n $i -s $stepSize1 -f $stepSize2 -g $stepSize3  
done
