# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from traffic import Traffic, TrafficMultiClass, GTSDBMultiClass

"""Factory method for easily getting imdbs by name."""

__sets = {}


# Set up traffic
# path to devkit example '/home/szy/INRIA'
traffic_devkit_path = "/mnt/nvme/traffic-rcnn/"
traffic_multi_class_devkit_path = "/mnt/nvme/traffic-multiclass/"
traffic_dual_devkit_path = "/mnt/nvme/traffic-dual-purpose/"
gtsrb_devkit_path = "/mnt/nvme/gtsrb-rcnn/"
gtsdb_devkit_path = "/mnt/nvme/gtsdb-rcnn/"
gtsdb_multi_class_devkit_path = "/mnt/nvme/gtsdb-rcnn-multi/"

for split in ['train', 'test', 'test2']:
    name = '{}_{}'.format('traffic', split)
    __sets[name] = (
        lambda split=split: Traffic(split, traffic_devkit_path))

for split in ['train', 'test']:
    name = '{}_{}'.format('traffic_multi_class', split)
    __sets[name] = (
        lambda split=split: TrafficMultiClass(
            split, traffic_multi_class_devkit_path))

for split in ['train', 'test', 'test2']:
    name = '{}_{}'.format('traffic_dual', split)
    __sets[name] = (
        lambda split=split: Traffic(
            split, traffic_dual_devkit_path))

for split in ['train', 'test']:
    name = '{}_{}'.format('gtsrb', split)
    __sets[name] = (
        lambda split=split: Traffic(split, gtsrb_devkit_path))

for split in ['train', 'test', 'full']:
    name = '{}_{}'.format('gtsdb', split)
    __sets[name] = (
        lambda split=split: Traffic(split, gtsdb_devkit_path))

for split in ['train', 'test', 'full']:
    name = '{}_{}'.format('gtsdb_multi_class', split)
    __sets[name] = (
        lambda split=split: GTSDBMultiClass(
            split, gtsdb_multi_class_devkit_path))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
