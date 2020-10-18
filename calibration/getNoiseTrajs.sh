#! /bin/bash
tasks=('left_turn') 
Intersections=(70_63)
min_frames=(45) 
min_frames_noise=(63)
max_frames=(90) 
adversary_names=('adversary_') 
townName='Town01_nemesisA'
portNum=2000
steerNoise=(2 3 4)
trajectoryNum=(2 3 4) 
for i in "${!steerNoise[@]}";do
    ## If you want to define the random noise yourself, please uncomment the two lines below.
    #cp -r datasets_1/${tasks[0]}/${Intersections[0]} datasets_$trajectoryNum/${tasks[0]}
    #python get_steer_selfCorrection.py -n ${steerNoise[i]} -j $trajectoryNum -t ${tasks[0]} -i ${Intersections[0]} -a ${min_frames_noise[0]} -b ${max_frames[0]}
    ./runGetFrames.sh ${tasks[0]} ${Intersections[0]} ${min_frames[0]} ${max_frames[0]} ${adversary_names[0]} $townName $portNum $trajectoryNum
    ./after.sh ${tasks[0]} ${Intersections[0]} ${min_frames[0]} ${max_frames[0]} ${adversary_names[0]} $townName $portNum $trajectoryNum
    trajectoryNum=$((trajectoryNum+1))
done
for i in "${!steerNoise[@]}";do
    ./getHomographyMat.sh ${tasks[0]} ${Intersections[0]} ${min_frames[0]} ${max_frames[0]} ${trajectoryNum[i]}
    trajectoryNum=$((trajectoryNum+1))
done
