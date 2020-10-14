#! /bin/bash

task=${1?Erro: no task}
intersection=${2?Error: no intersection}
minFrame=${3?Error: no min frame input}
maxFrame=${4?Error: no max frame input}
trajectoryNum=${5?Error: trajectory number}
python homographyMat_caculator/frames_h_save.py -t $task -i $intersection -l $minFrame -z $maxFrame -j $trajectoryNum
python homographyMat_caculator/run.py -t $task -i $intersection -l $minFrame -z $maxFrame -j $trajectoryNum

