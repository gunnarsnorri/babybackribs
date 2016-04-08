#!/usr/bin/env python

import os
import os.path
import shutil
import argparse


def line_to_path(line):
    line = line.strip("\n")
    image_path = os.path.join(line.split(" ")[0]+line.split(" ")[1])
    tmp = line.split(" ")[1]
    class_name = tmp.split("/")[0]
    image_name = tmp.split("/")[1]
    return image_path, class_name, image_name

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=True, help="Path to train.txt")
    ap.add_argument("-v", "--val", required=True, help="Path to train.txt")
    ap.add_argument("-o", "--outdir", required=True,
                    help="Path to output directory")
    args = vars(ap.parse_args())

    train_path = args["train"]
    val_path = args["val"]
    data_dir = args["outdir"]
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    with open(train_path) as read_f:
        for line in read_f:
            image_path, class_name, image_name = line_to_path(line)
            image_output = os.path.join(train_dir, class_name, image_name)
            if not os.path.exists(os.path.join(train_dir, class_name)):
                os.mkdir(os.path.join(train_dir, class_name))
            shutil.copy(image_path, image_output)
