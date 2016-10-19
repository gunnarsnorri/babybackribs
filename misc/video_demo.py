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
import numpy as np
import caffe
import os
import cv2
import argparse

from googlenet_deploy import GoogLeNet

NETS = {
    'pascal_voc':
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
         'tvmonitor'),
    'traffic':
            (
            '__background__',
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
            'URDBL'
        ),
    'gtsdb':
        (
            '__background__',
            '20_SIGN',
            '30_SIGN',
            '50_SIGN',
            '60_SIGN',
            '70_SIGN',
            '80_SIGN',
            '80_ENDS',
            '100_SIGN',
            '120_SIGN',
            'NO_OVERTAKING',
            'NO_OVERTAKING_TRUCKS',
            'RIGHT_OF_WAY',
            'PRIORITY_ROAD',
            'GIVE_WAY',
            'STOP',
            'NO_VEHICLES',
            'NO_TRUCKS',
            'NO_ENTRY',
            'CAUTION',
            'DANGER_CURVE_LEFT',
            'DANGER_CURVE_RIGHT',
            'DOUBLE_CURVE_LEFT',
            'BUMPY_ROAD',
            'SLIPPERY_ROAD',
            'NARROWS_RIGHT',
            'ROADWORKS',
            'TRAFFIC_SIGNALS',
            'PEDESTRIANS',
            'CHILDREN',
            'BICYCLES',
            'SNOW',
            'WILD_ANIMALS',
            'END_SPEED_LIMITS',
            'TURN_RIGHT',
            'TURN_LEFT',
            'AHEAD_ONLY',
            'STRAIGHT_OR_RIGHT',
            'STRAIGHT_OR_LEFT',
            'KEEP_RIGHT',
            'KEEP_LEFT',
            'ROUNDABOUT',
            'END_NO_OVERTAKING',
            'END_NO_OVERTAKING_TRUCKS'
        ),
    'single': ('__background__', 'sign')
}


def get_detections(class_name, dets, thresh=0.5):
    """Get detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    bboxes = [dets[i, :4] for i in inds]
    scores = [dets[i, -1] for i in inds]
    return bboxes, scores


def video_demo(net, im, classes, googlenet):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.01
    all_bboxes = []
    all_scores = []
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        conf_boxes, conf_scores = get_detections(cls, dets, CONF_THRESH)
        all_bboxes.append(conf_boxes)
        all_scores.append(conf_scores)
    if googlenet is not None:
        new_all_bboxes = [[] for i in range(len(googlenet.classes))]
        new_all_scores = [[] for i in range(len(googlenet.classes))]
        if any(all_bboxes):
            flat_confidence = np.array([score for cls_score in all_scores
                            for score in cls_score])
            flat_boxes = np.array([box for cls_box in all_bboxes for box in cls_box])
            new_classifications = googlenet.classify(
                im, flat_confidence, flat_boxes, 0)
            for i, classification in enumerate(new_classifications["prob"]):
                class_index = classification.argmax()
                new_score = classification.max()
                if new_score >= CONF_THRESH:
                    new_all_bboxes[class_index].append(flat_boxes[i])
                    new_all_scores[class_index].append(new_score)
        return new_all_bboxes, new_all_scores
    return all_bboxes, all_scores


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
    parser.add_argument(
        '--prototxt',
        dest='prototxt',
        help='Prototxt file to use',
     required=True)
    parser.add_argument('--model', dest='demo_model', help='Model to use',
                        required=True)
    parser.add_argument('--video', dest='video', help='Input video file',
                        required=True)
    parser.add_argument('--out', dest='out', help='Output .avi video file',
                        required=True)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--googlenet', dest='googlenet',
                        help='Flag whether GoogLeNet is used or not',
                        action='store_true')
    parser.set_defaults(googlenet=False)

    args = parser.parse_args()

    return args


def draw_on_frame(frame, boxes, scores, classes):
    for cls_ind, cls in enumerate(classes):
        for box_ind, bbox in enumerate(boxes[cls_ind]):
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 1
            textsize, _ = cv2.getTextSize(cls, font, font_scale, font_thickness)
            cv2.putText(
                frame,
                cls,
                ((x1 + x2 - textsize[0])/2, y1),
                font, font_scale, (255, 255, 255),
                font_thickness, cv2.CV_AA)
            score = str(scores[cls_ind][box_ind])
            textsize, _ = cv2.getTextSize(
                score, font, font_scale, font_thickness)
            cv2.putText(
                frame,
                score,
                ((x1 + x2 - textsize[0])/2, y2 + textsize[1]),
                font, font_scale, (255, 255, 255),
                font_thickness, cv2.CV_AA)


if __name__ == '__main__':
    args = parse_args()

    prototxt = args.prototxt
    classes = NETS[args.demo_net]
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

    # Load GoogLeNet if needed
    if args.googlenet:
        # g_prototxt = "/mnt/nvme/caffe/models/bvlc_googlenet/deploy.prototxt"
        # g_caffemodel = "/mnt/nvme/caffe/models/bvlc_googlenet/bvlc_googlenet_tune_iter_70000.caffemodel"
        # label_file = "/usr/share/digits/digits/jobs/20160530-164726-bf90/labels.txt"
        g_prototxt = "/mnt/nvme/caffe_models/deploy.prototxt"
        g_caffemodel = "/mnt/nvme/caffe_models/bvlc_googlenet_gtsrb_iter_46590.caffemodel"
        label_file = "/mnt/nvme/caffe_models/gtsdb_text_labels.txt"
        with open(label_file, 'r') as lf:
            label_lines = lf.readlines()
        g_classes = [x.strip().replace(" ", "_") for x in label_lines]
        # mean_file = "/mnt/nvme/caffe/models/bvlc_googlenet/mean.npy"
        mean_file = "/mnt/nvme/caffe_models/gtsrb_mean.npy"
        g_mean = np.load(mean_file).mean(1).mean(1)
        googlenet = GoogLeNet(
            g_prototxt,
            g_caffemodel,
            g_classes,
            g_mean,
            args.gpu_id)
    else:
        googlenet = None

    # Open video with opencv
    video_capture = cv2.VideoCapture(args.video)
    num_frames = video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    ret, frame = video_capture.read()
    height, width, _ = frame.shape
    timer = Timer()

    out = cv2.VideoWriter(
        args.out, cv2.cv.CV_FOURCC(*'XVID'),
        fps, (width, height))

    t = 0
    time_since_calc = 0
    frame_count = 0
    classes_for_draw = classes[1:] if googlenet is None else googlenet.classes
    while ret:
        frame_count += 1
        if time_since_calc >= t:  # to run in real time compared to video
            print("Processing frame %d/%d (%f)" % (frame_count, num_frames, t))
            timer.tic()
            boxes_by_class, scores_by_class = video_demo(
                net, frame, classes, googlenet)
            current_boxes = boxes_by_class
            current_scores = scores_by_class
            t = timer.toc(False)
            time_since_calc = 0
        draw_on_frame(frame, current_boxes, current_scores, classes_for_draw)
        out.write(frame)
        time_since_calc += 1/fps

        ret, frame = video_capture.read()
    out.release()
    video_capture.release()
