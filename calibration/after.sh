#! /bin/bash

task=${1?Erro: no task}
intersection=${2?Error: no intersection}
minFrame=${3?Error: no min frame input}
maxFrame=${4?Error: no max frame input}
adversaryName=${5?Erro: adversary name}
townName=${6?Error: no town name}
portNum=${7?Error: port nubmer}
trajectoryNum=${8?Error: trajectory number}

echo $task
echo $intersection
echo $minFrame
echo $maxFrame

color=True
echo $task
echo $intersection

python carla_env_after_homographyMat.py -r $portNum -n $townName -p 0 -s $intersection -t $task -c $color -w $minFrame -z $maxFrame -a $adversaryName -u 1 -j $trajectoryNum
