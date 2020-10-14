if echo $(sudo docker --version) | grep -Eq '19.0[3-9]|19.[1-9][0-9]|20' ; then
    echo "Docker version >= 19.03 detected...";
    sudo docker run \
      -it \
      --gpus 0 \
      --net="host" \
      jinghan20/carla:v3\
      /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisA \
      -benchmark \
      -carla-server \
      -fps=10 \
      -world-port=2000 \
      -windowed -ResX=100 -ResY=100 \
      -carla-no-hud;
else
    echo "Docker version < 19.03 detected...";
    sudo docker run \
      -it \
      --runtime=nvidia \
      --net="host" \
      jinghan20/carla:v3 \
      /bin/bash CarlaUE4.sh /Game/Maps/Town01_nemesisA \
      -e NVIDIA_VISIBLE_DEVICES=0 \
      -benchmark \
      -carla-server \
      -fps=10 \
      -world-port=2000 \
      -windowed -ResX=100 -ResY=100 \
      -carla-no-hud;
fi
