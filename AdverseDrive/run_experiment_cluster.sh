screen -dmS client_Town01_nemesisA bash -c 'sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes --name Town01_nemesisA xzgroup/adversedrive:latest python bay_4.py -c 1.json > 4.txt'

screen -dmS client_Town01_nemesisB bash -c 'sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -e CUDA_VISIBLE_DEVICES=1 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes --name Town01_nemesisB xzgroup/adversedrive:latest python bay_7.py -c 2.json > 7.txt'

screen -dmS client_Town01_nemesisC bash -c 'sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=2 -e CUDA_VISIBLE_DEVICES=2 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes --name Town01_nemesisC xzgroup/adversedrive:latest python bay_14.py -c 3.json > 14.txt'

screen -dmS client_Town01_nemesisD bash -c 'sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=3 -e CUDA_VISIBLE_DEVICES=3 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes --name Town01_nemesisD xzgroup/adversedrive:latest python bay_21.py -c 4.json > 21.txt'

screen -dmS client_Town01_nemesisE bash -c 'sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=4 -e CUDA_VISIBLE_DEVICES=4 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes --name Town01_nemesisE xzgroup/adversedrive:latest python bay_28.py -c 5.json > 28.txt'

screen -dmS client_Town01_nemesisF bash -c 'sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 -e CUDA_VISIBLE_DEVICES=5 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes --name Town01_nemesisF xzgroup/adversedrive:latest python bay_35.py -c 6.json > 35.txt'

#screen -dmS client_Town01_nemesisG bash -c 'sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=6 -e CUDA_VISIBLE_DEVICES=6 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes --name Town01_nemesisF xzgroup/adversedrive:latest python bay_35.py -c 6.json'

#screen -dmS client_Town01_nemesisH bash -c 'sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=7 -e CUDA_VISIBLE_DEVICES=7 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes --name Town01_nemesisF xzgroup/adversedrive:latest python bay_35.py -c 6.json'
