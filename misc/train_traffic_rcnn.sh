#!/bin/bash



STANDARD_ARGS="--iters 70000 --cfg cfgs/faster_rcnn_end2end_original.yml --weights /mnt/nvme/py-faster-rcnn-3/data/imagenet_models/VGG16.v2.caffemodel"


# ./train_traffic_net.py --solver models/VGG16_end2end_original/solver.prototxt --imdb traffic_train --run-name traffic_with_imagenet_original $STANDARD_ARGS
# ./train_traffic_net.py --solver models/VGG16_end2end/solver.prototxt --imdb traffic_train --run-name traffic_with_imagenet $STANDARD_ARGS
./train_traffic_net.py --solver models/VGG16_end2end/solver.prototxt --imdb gtsrb_train --run-name gtsrb_with_imagenet $STANDARD_ARGS
./train_traffic_net.py --solver models/VGG16_end2end_original/solver.prototxt --imdb gtsrb_train --run-name gtsrb_with_imagenet_original $STANDARD_ARGS
