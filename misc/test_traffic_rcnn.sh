#!/bin/bash

# ALL_ITERS="10000 20000 30000 40000 50000 60000 70000"
ALL_ITERS="70000"

DIR="/mnt/nvme/caffe_results/vgg16_min_size_4_multi"
mkdir -p $DIR
echo "Starting 4 multi"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/VGG16_end2end_multi_class/test.prototxt  --net /mnt/nvme/caffe_output/vgg16_min_size_4_multi/train/vgg16_faster_rcnn_multiclass_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_new_multi.yml --imdb traffic_multi_class_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/vgg16_min_size_4_single"
mkdir -p $DIR
echo "Starting 4 single"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/VGG16_end2end/test.prototxt  --net /mnt/nvme/caffe_output/vgg16_min_size_4_single/train/vgg16_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end.yml --imdb traffic_test  > ${DIR}/${ITERS}.txt

done


DIR="/mnt/nvme/caffe_results/vgg16_min_size_16_multi"
mkdir -p $DIR
echo "Starting 16 multi"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/VGG16_end2end_multi_class/test.prototxt  --net /mnt/nvme/caffe_output/vgg16_min_size_16_multi/train/vgg16_faster_rcnn_multiclass_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_multi_class.yml --imdb traffic_multi_class_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/vgg16_min_size_16_single"
mkdir -p $DIR
echo "Starting 16 single"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/VGG16_end2end/test.prototxt  --net /mnt/nvme/caffe_output/vgg16_min_size_16_single/train/vgg16_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_original.yml --imdb traffic_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/zf_min_size_4_multi"
mkdir -p $DIR
echo "Starting 4 multi"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/ZF_end2end_multi_class/test.prototxt  --net /mnt/nvme/caffe_output/zf_min_size_4_multi/train/zf_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_new_multi.yml --imdb traffic_multi_class_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/zf_min_size_4_single"
mkdir -p $DIR
echo "Starting 4 single"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/ZF_end2end/test.prototxt  --net /mnt/nvme/caffe_output/zf_min_size_4_single/train/zf_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end.yml --imdb traffic_test  > ${DIR}/${ITERS}.txt

done


DIR="/mnt/nvme/caffe_results/zf_min_size_16_multi"
mkdir -p $DIR
echo "Starting 16 multi"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/ZF_end2end_multi_class/test.prototxt  --net /mnt/nvme/caffe_output/zf_min_size_16_multi/train/zf_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_multi_class.yml --imdb traffic_multi_class_test  > ${DIR}/${ITERS}.txt
done



DIR="/mnt/nvme/caffe_results/zf_min_size_16_single"
mkdir -p $DIR
echo "Starting 16 single"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/ZF_end2end/test.prototxt  --net /mnt/nvme/caffe_output/zf_min_size_16_single/train/zf_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_original.yml --imdb traffic_test  > ${DIR}/${ITERS}.txt

done









DIR="/mnt/nvme/caffe_results/gtsdb_vgg16_min_size_4_multi"
mkdir -p $DIR
echo "Starting 4 multi"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/VGG16_end2end_gtsdb/test.prototxt  --net /mnt/nvme/caffe_output/gtsdb_vgg16_min_size_4_multi/train/vgg16_faster_rcnn_multiclass_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_new_multi.yml --imdb gtsdb_multi_class_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/gtsdb_vgg16_min_size_4_single"
mkdir -p $DIR
echo "Starting 4 single"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/VGG16_end2end/test.prototxt  --net /mnt/nvme/caffe_output/gtsdb_vgg16_min_size_4_single/train/vgg16_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end.yml --imdb gtsdb_test  > ${DIR}/${ITERS}.txt

done


DIR="/mnt/nvme/caffe_results/gtsdb_vgg16_min_size_16_multi"
mkdir -p $DIR
echo "Starting 16 multi"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/VGG16_end2end_gtsdb/test.prototxt  --net /mnt/nvme/caffe_output/gtsdb_vgg16_min_size_16_multi/train/vgg16_faster_rcnn_multiclass_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_multi_class.yml --imdb gtsdb_multi_class_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/gtsdb_vgg16_min_size_16_single"
mkdir -p $DIR
echo "Starting 16 single"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/VGG16_end2end/test.prototxt  --net /mnt/nvme/caffe_output/gtsdb_vgg16_min_size_16_single/train/vgg16_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_original.yml --imdb gtsdb_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/gtsdb_zf_min_size_4_multi"
mkdir -p $DIR
echo "Starting 4 multi"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/ZF_end2end_gtsdb/test.prototxt  --net /mnt/nvme/caffe_output/gtsdb_zf_min_size_4_multi/train/zf_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_new_multi.yml --imdb gtsdb_multi_class_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/gtsdb_zf_min_size_4_single"
mkdir -p $DIR
echo "Starting 4 single"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/ZF_end2end/test.prototxt  --net /mnt/nvme/caffe_output/gtsdb_zf_min_size_4_single/train/zf_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end.yml --imdb gtsdb_test  > ${DIR}/${ITERS}.txt

done


DIR="/mnt/nvme/caffe_results/gtsdb_zf_min_size_16_multi"
mkdir -p $DIR
echo "Starting 16 multi"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/ZF_end2end_gtsdb/test.prototxt  --net /mnt/nvme/caffe_output/gtsdb_zf_min_size_16_multi/train/zf_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_multi_class.yml --imdb gtsdb_multi_class_test  > ${DIR}/${ITERS}.txt

done



DIR="/mnt/nvme/caffe_results/gtsdb_zf_min_size_16_single"
mkdir -p $DIR
echo "Starting 16 single"
for ITERS in $ALL_ITERS
do
    echo "Model with $ITERS iterations starting"
    ./test_traffic_net.py --def models/ZF_end2end/test.prototxt  --net /mnt/nvme/caffe_output/gtsdb_zf_min_size_16_single/train/zf_faster_rcnn_iter_${ITERS}.caffemodel --cfg cfgs/faster_rcnn_end2end_original.yml --imdb gtsdb_test  > ${DIR}/${ITERS}.txt

done
