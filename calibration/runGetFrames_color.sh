#! /bin/bash

task=${1?Erro: no task}
intersection=${2?Error: no intersection}
minFrame=${3?Error: no min frame input}
maxFrame=${4?Error: no max frame input}
adversaryName=${5?Erro: adversary name}
townName=${6?Error: no town name}

echo $task
echo $intersection
echo $minFrame
echo $maxFrame


color=False   
echo $task
echo $intersection

for i in {0..22}
do
	echo $i
	python carla_env_color.py -n $townName -p $i -s $intersection -t $task -c $color -w $minFrame -z $maxFrame -a $adversaryName -u 1
done

