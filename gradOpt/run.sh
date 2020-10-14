#! /bin/bash
stepSize1=${1?Error: no stepSize}
stepSize2=${2?Error: no stepSize}
stepSize3=${3?Error: no stepSize}
./run_one_intersection.sh left_turn 70_63 $stepSize1 $stepSize2 $stepSize3 
./run_one_intersection4.sh left_turn 70_63 $stepSize1 $stepSize2 $stepSize3 
