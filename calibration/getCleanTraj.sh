#! /bin/bash
tasks=('left_turn') 
Intersections=(70_63)
min_frames=(35) 
min_frames_noise=(45)
max_frames=(65) 
adversary_names=('adversary_') 
townName='Town01_nemesisA'
portNum=2000
steerNoise=(1)
trajectoryNum=(1)

## collect data for homography matrix calculation
for i in "${!steerNoise[@]}";do
    ./find_intersection_numbers_batch.sh ${tasks[0]} ${Intersections[0]} ${min_frames[0]} ${max_frames[0]} ${adversary_names[0]} $townName $portNum $trajectoryNum
    ./runGetFrames.sh ${tasks[0]} ${Intersections[0]} ${min_frames[0]} ${max_frames[0]} ${adversary_names[0]} $townName $portNum $trajectoryNum
    #./after.sh ${tasks[0]} ${Intersections[0]} ${min_frames[0]} ${max_frames[0]} ${adversary_names[0]} $townName $portNum $trajectoryNum
    trajectoryNum=$((trajectoryNum+1))
done

trajectoryNum=(1)
## calculate homography matrix
for i in "${!steerNoise[@]}";do
    ./getHomographyMat.sh ${tasks[0]} ${Intersections[0]} ${min_frames[0]} ${max_frames[0]} ${trajectoryNum[i]}
    trajectoryNum=$((trajectoryNum+1))
done

