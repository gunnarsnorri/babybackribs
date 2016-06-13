#!/usr/bin/env python
import sys
import numpy as np
import argparse

from googlenet_deploy import GoogLeNet
import _init_paths
import factory
from utils.timer import Timer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        required=True, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        required=True, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='traffic_dual_test', type=str)
    parser.add_argument('--labels', dest='labels_file',
                        help='Labels file',
                        required=True, type=str)
    parser.add_argument('--det', dest='detfile',
                        help='File containing region proposals',
                        required=True, type=str)
    parser.add_argument('--mean', dest='meanfile',
                        help='File containing mean',
                        required=True, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    t = Timer()
    with open(args.labels_file, 'r') as lf:
        label_lines = lf.readlines()
    classes = [x.strip().replace(" ", "_") for x in label_lines]
    mean = np.load(args.meanfile).mean(1).mean(1)
    g = GoogLeNet(
        args.prototxt,
        args.caffemodel,
        classes, mean, gpu_id = args.gpu_id)
    print(mean)
    traffic_imdb = factory.get_imdb(args.imdb_name)
    with open(args.detfile, 'r') as df:
        det_lines = df.readlines()
    splitlines = [x.strip().split(' ') for x in det_lines]
    det_by_img = {x[0]: x[1:] for x in splitlines}

    det_by_img = {}
    for x in splitlines:
        if not x[0] in det_by_img:
            det_by_img[x[0]] = []
        det_by_img[x[0]].append(x[1:])

    count = 0
    all_boxes = [
        [[] for _ in xrange(len(traffic_imdb.image_index))]
        for _ in xrange(len(g.classes) + 1)]  # +1 due to background
    for image_index, dets in det_by_img.iteritems():
        confidence = [d[0] for d in dets]
        boxes = np.array(
            [[float(z) for z in d[1:5]] for d in dets])
        image_fullpath = traffic_imdb.image_path_from_index(image_index)
        t.tic()
        img_classifications = g.classify(image_fullpath, confidence, boxes, 0.0)
        average_time = t.toc()
        classes = [g.classes[c.argmax()] for c in img_classifications["prob"]]
        count += 1
        if count % 100 == 1:
            print("Classifying image {:d}/{:d} ({:f})".
                  format(count, traffic_imdb.num_images, average_time))

        new_dets = []
        for i, classification in enumerate(img_classifications["prob"]):
            class_index = classification.argmax() + 1  # +1 due to background
            image_number = traffic_imdb.image_index.index(image_index)
            new_det = np.append(boxes[i], classification.max())
            if len(all_boxes[class_index][image_number]) == 0:
                all_boxes[class_index][image_number] = new_det[None, :]
            else:
                all_boxes[class_index][image_number] = np.concatenate(
                    (all_boxes[class_index][image_number], new_det[None, :]), 0)
                print(all_boxes[class_index][image_number])

    traffic_imdb._classes = g.classes
    traffic_imdb._classes.insert(0, "__background__")
    traffic_imdb.is_googlenet = True
    traffic_imdb.evaluate_detections(all_boxes, "/mnt/nvme/googlenet_results")
