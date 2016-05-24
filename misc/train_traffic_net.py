#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import sys
import os.path
import os
import shutil
import yaml
import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--run-name', dest='run_name',
                        help='Name of output dir folder',
                        default="", type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


def get_output_dir(cfg, imdb, run_name):
    if cfg.TRAIN.OUTPUT_DIR is not None:
        output_dir = cfg.TRAIN.OUTPUT_DIR
    else:
        output_dir = os.path.join("/mnt/nvme/caffe_output", run_name, imdb.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def get_train_file(solver):
    with open(solver, "r") as solver_file:
        for line in solver_file:
            if line.startswith("train_net:"):
                train_filename = line.strip("\n").split(":")[1]
                train_filename = train_filename.strip().strip('"')
                return train_filename
    raise KeyError("train_net not found in %s" % solver)



if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    cfg.TRAIN.OUTPUT_DIR = None
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    cache_file = os.path.abspath(
        os.path.join(cfg.DATA_DIR, 'cache',
                     args.imdb_name.split("_")[-1] + '_gt_roidb.pkl'))
    if os.path.exists(cache_file):
        print("Removing cache file : %s" % cache_file)
        os.remove(cache_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb, roidb = combined_roidb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(cfg, imdb, args.run_name)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # Write relevant information to output_dir for documentation
    with open(os.path.join(output_dir, "config.yml"), "w") as conf_file:
        conf_file.write(yaml.dump(cfg))
    shutil.copy(args.solver, output_dir)
    train_file = get_train_file(args.solver)
    shutil.copy(train_file, output_dir)
    with open(os.path.join(output_dir, "extra_info.txt"), "w") as extra_file:
        extra_file.write("%s\n" % args.pretrained_model)
        extra_file.write("%s\n" % args.imdb_name)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
