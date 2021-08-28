# !/bin/bash

SOURCE=${1:-$(pwd)}
DATASETS=${2:-"/media/serlini/data/Datasets/"}

echo $(echo "Hello $(whoami)")

docker run --rm -it --gpus all \
        -v /dev/shm:/dev/shm \
        -v $SOURCE:/home/yolo9000/:rw \
        -v $DATASETS:/datasets/:ro \
        --name mindspore \
        --privileged=true \
        mindspore/mindspore-gpu:1.3.0