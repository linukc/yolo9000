# !/bin/bash

echo $(echo "Hello $(whoami)")

docker exec -ti yolo9000_container /bin/bash