#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import caffe
import os
import cv2
import argparse

NETS = {
    'pascal_voc': (
        '/mnt/nvme/py-faster-rcnn-3/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt',
        ('__background__',
         'aeroplane',
         'bicycle',
         'bird',
         'boat',
         'bottle',
         'bus',
         'car',
         'cat',
         'chair',
         'cow',
         'diningtable',
         'dog',
         'horse',
         'motorbike',
         'person',
         'pottedplant',
         'sheep',
         'sofa',
         'train',
         'tvmonitor')),
    'traffic': (
            'models/VGG16_end2end/test.prototxt',
            ('__background__',
             'sign')),
    'traffic_original': (
            'models/VGG16_end2end_original/test.prototxt',
            ('__background__',
             'sign')),
    'traffic_multi_class': (
            'models/VGG16_end2end_multi_class/test.prototxt',
            ('__background__',
             '30_SIGN',
             '50_SIGN',
             '60_SIGN',
             '70_SIGN',
             '80_SIGN',
             '90_SIGN',
             '100_SIGN',
             '110_SIGN',
             '120_SIGN',
             'GIVE_WAY',
             'NO_PARKING',
             'NO_STOPPING_NO_STANDING',
             'OTHER',
             'PASS_EITHER_SIDE',
             'PASS_LEFT_SIDE',
             'PASS_RIGHT_SIDE',
             'PEDESTRIAN_CROSSING',
             'PRIORITY_ROAD',
             'STOP',
             'URDBL')),
    'alt_opt': (
            'models/VGG16_alt_opt/faster_rcnn_test.pt',
            ('__background__',
             'sign')),
    'ZF': (
            'models/ZF_end2end/test.prototxt',
            ('__background__',
             'sign')),
    'gtsdb_multi': (
            'models/ZF_end2end_gtsdb/test.prototxt',
            ('__background__',) + tuple([str(i) for i in range(43)]))}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.01
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--model', dest='demo_model', help='Model to use',
                        required=True)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    prototxt = NETS[args.demo_net][0]
    classes = NETS[args.demo_net][1]
    caffemodel = args.demo_model

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    # im_dir = "/mnt/nvme/test_pics"
    # im_names = ["kors-%d.png" % i for i in range(1, 5)] + ["img.jpg"]
    # im_names = [os.path.join(im_dir, img) for img in im_names]

    im_dir = "/mnt/nvme/gtsdb-rcnn/data/Images/"
    im_names = ["00%d.ppm" % i for i in range(770, 777)]
    im_names = [os.path.join(im_dir, img) for img in im_names]
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name, classes)

    plt.show()
