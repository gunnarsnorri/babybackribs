#!/bin/bash



# Multi class ZF
./train_traffic_net.py --solver models/ZF_end2end_gtsdb/solver.prototxt --imdb gtsdb_multi_class_train --run-name gtsdb_zf_min_size_4_multi --cfg cfgs/faster_rcnn_end2end_new_multi.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel

./train_traffic_net.py --solver models/ZF_end2end_gtsdb/solver.prototxt --imdb gtsdb_multi_class_train --run-name gtsdb_zf_min_size_16_multi --cfg cfgs/faster_rcnn_end2end_multi_class.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel

# Multi class VGG
./train_traffic_net.py --solver models/VGG16_end2end_gtsdb/solver.prototxt --imdb gtsdb_multi_class_train --run-name gtsdb_vgg16_min_size_4_multi --cfg cfgs/faster_rcnn_end2end_new_multi.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel

./train_traffic_net.py --solver models/VGG16_end2end_gtsdb/solver.prototxt --imdb gtsdb_multi_class_train --run-name gtsdb_vgg16_min_size_16_multi --cfg cfgs/faster_rcnn_end2end_multi_class.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel

# Single class ZF
./train_traffic_net.py --solver models/ZF_end2end/solver.prototxt --imdb gtsdb_train --run-name gtsdb_zf_min_size_4_single --cfg cfgs/faster_rcnn_end2end.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel

./train_traffic_net.py --solver models/ZF_end2end/solver.prototxt --imdb gtsdb_train --run-name gtsdb_zf_min_size_16_single --cfg cfgs/faster_rcnn_end2end_original.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel

# Single class VGG
./train_traffic_net.py --solver models/VGG16_end2end/solver.prototxt --imdb gtsdb_train --run-name gtsdb_vgg16_min_size_4_single --cfg cfgs/faster_rcnn_end2end.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel

./train_traffic_net.py --solver models/VGG16_end2end/solver.prototxt --imdb gtsdb_train --run-name gtsdb_vgg16_min_size_16_single --cfg cfgs/faster_rcnn_end2end_original.yml --iters 70000 --weights /mnt/nvme/py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel
