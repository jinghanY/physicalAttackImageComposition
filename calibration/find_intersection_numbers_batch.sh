#! /bin/bash

task=${1?Error: no task}
intersection=${2?Error: no intersection}
minFrame=${3?Error: no min frame input}
maxFrame=${4?Error: no max frame input}
adversaryName=${5?Error: no adversary name}
townName=${6?Error: no town name}
portNum=${7?Error: port nubmer}
trajectoryNum=${8?Error: trajectory nubmer}
python create_datasets.py -t $task -i $intersection -j $trajectoryNum 
color=True
python carla_env_before_homographyMat.py -r $portNum -n $townName -p 0 -s $intersection -t $task -c $color -v True -w $minFrame -z $maxFrame -a $adversaryName -j $trajectoryNum
