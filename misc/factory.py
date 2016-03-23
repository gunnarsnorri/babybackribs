# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from traffic import Traffic

"""Factory method for easily getting imdbs by name."""

__sets = {}


# Set up traffic
# path to devkit example '/home/szy/INRIA'
traffic_devkit_path = "/mnt/nvme/traffic-rcnn/"

for split in ['train', 'test']:
    name = '{}_{}'.format('traffic', split)
    __sets[name] = (
        lambda split=split: Traffic(
            split, traffic_devkit_path))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
