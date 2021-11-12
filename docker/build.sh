docker build --build-arg UID=$(id -u) \
             --build-arg GID=$(id -g) \
             --tag yolo9000 .