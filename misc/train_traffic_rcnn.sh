#!/bin/bash



# STANDARD_ARGS="--iters 70000 --weights /mnt/nvme/py-faster-rcnn-3/data/imagenet_models/VGG16.v2.caffemodel"


# ./train_traffic_net.py --solver models/VGG16_end2end_original/solver.prototxt --imdb traffic_train --run-name traffic_with_imagenet_original $STANDARD_ARGS
# ./train_traffic_net.py --solver models/VGG16_end2end/solver.prototxt --imdb traffic_train --run-name traffic_with_imagenet $STANDARD_ARGS
# ./train_traffic_net.py --solver models/VGG16_end2end/solver.prototxt --imdb gtsrb_train --run-name gtsrb_with_imagenet $STANDARD_ARGS
# ./train_traffic_net.py --solver models/VGG16_end2end_original/solver.prototxt --imdb gtsrb_train --run-name gtsrb_with_imagenet_original $STANDARD_ARGS

# ./train_traffic_net.py --solver models/VGG16_end2end/solver.prototxt --imdb traffic_train --run-name min_size_single --cfg cfgs/faster_rcnn_end2end.yml $STANDARD_ARGS
# ./train_traffic_net.py --solver models/VGG16_end2end_multi_class/solver.prototxt --imdb traffic_multi_class_train --run-name min_size_multi --cfg cfgs/faster_rcnn_end2end_new_multi.yml $STANDARD_ARGS

./train_traffic_net.py --solver models/ZF_end2end_multi_class/solver.prototxt --imdb traffic_multi_class_train --run-name zf_min_size_4_multi --cfg cfgs/faster_rcnn_end2end_new_multi.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel

./train_traffic_net.py --solver models/ZF_end2end/solver.prototxt --imdb traffic_train --run-name zf_min_size_16_single --cfg cfgs/faster_rcnn_end2end_original.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel

