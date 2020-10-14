#!/bin/bash
screen -dmS Town01_nemesisA bash -c 'sudo docker run -i -t -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --net="host" --name carla_Town01_nemesisA jinghan20/carla:v3 /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisA -benchmark -carla-server -fps=10 -world-port=2000 -windowed -ResX=100 -ResY=100 -carla-no-hud; exec bash'

screen -dmS Town01_nemesisB bash -c 'sudo docker run -i -t -p 2100-2102:2100-2102 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 --net="host" --name carla_Town01_nemesisB jinghan20/carla:v3 /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisB -benchmark -carla-server -fps=10 -world-port=2100 -windowed -ResX=100 -ResY=100 -carla-no-hud; exec bash'

screen -dmS Town01_nemesisC bash -c 'sudo docker run -i -t -p 2200-2202:2200-2202 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=2 --net="host" --name carla_Town01_nemesisC jinghan20/carla:v3 /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisC -benchmark -carla-server -fps=10 -world-port=2200 -windowed -ResX=100 -ResY=100 -carla-no-hud; exec bash'

screen -dmS Town01_nemesisD bash -c 'sudo docker run -i -t -p 2300-2302:2300-2302 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=3 --net="host" --name carla_Town01_nemesisD jinghan20/carla:v3 /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisD -benchmark -carla-server -fps=10 -world-port=2300 -windowed -ResX=100 -ResY=100 -carla-no-hud; exec bash'

screen -dmS Town01_nemesisE bash -c 'sudo docker run -i -t -p 2400-2402:2400-2402 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=4 --net="host" --name carla_Town01_nemesisE jinghan20/carla:v3 /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisE -benchmark -carla-server -fps=10 -world-port=2400 -windowed -ResX=100 -ResY=100 -carla-no-hud; exec bash'

screen -dmS Town01_nemesisF bash -c 'sudo docker run -i -t -p 2500-2502:2500-2502 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 --net="host" --name carla_Town01_nemesisF jinghan20/carla:v3 /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisF -benchmark -carla-server -fps=10 -world-port=2500 -windowed -ResX=100 -ResY=100 -carla-no-hud; exec bash'

screen -dmS Town01_nemesisG bash -c 'sudo docker run -i -t -p 2600-2602:2600-2602 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=6 --net="host" --name carla_Town01_nemesisG jinghan20/carla:v3 /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisG -benchmark -carla-server -fps=10 -world-port=2600 -windowed -ResX=100 -ResY=100 -carla-no-hud; exec bash'

screen -dmS Town01_nemesisH bash -c 'sudo docker run -i -t -p 2700-2702:2700-2702 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=7 --net="host" --name carla_Town01_nemesisH jinghan20/carla:v3 /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisH -benchmark -carla-server -fps=10 -world-port=2700 -windowed -ResX=100 -ResY=100 -carla-no-hud; exec bash'

