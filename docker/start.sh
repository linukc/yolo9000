# !/bin/bash

SOURCE=${1:-$(pwd)}
DATASETS=${2:-"/media/serlini/data2/Datasets/"}

echo $(echo "Hello $(whoami)")

docker run --rm -it --gpus all \
        -v /dev/shm:/dev/shm \
        -v $SOURCE:/home/docker_yolo9000/yolo9000/:rw \
        -v $DATASETS:/datasets/:ro \
        --name yolo9000_container \
        yolo9000