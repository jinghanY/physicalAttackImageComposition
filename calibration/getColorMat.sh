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

## calculate color transform matrix
frameNumInput=45
for i in "${!tasks[@]}";do
	./runGetFrames_color.sh ${tasks[i]} ${Intersections[i]} ${min_frames[i]} ${max_frames[i]} ${adversary_names[i]} $townName $portNum
    python get_ccm.py -t ${tasks[i]} -i ${Intersections[i]} -f $frameNumInput
done
