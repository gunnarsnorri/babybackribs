#!/usr/bin/env python

import os
import os.path
import shutil
import argparse


def line_to_path(line):
    line = line.strip("\n")
    image_path = os.path.join(line.split(" ")[0])
    class_name = image_path.split("/")[-2]
    image_name = image_path.split("/")[-1]
    return image_path, class_name, image_name

def move_files(txt_path, dir_path):
    with open(txt_path) as read_f:
        for line in read_f:
            image_path, class_name, image_name = line_to_path(line)
            image_output = os.path.join(dir_path, class_name, image_name)
            if not os.path.exists(os.path.join(dir_path, class_name)):
                os.mkdir(os.path.join(dir_path, class_name))
            shutil.copy(image_path, image_output)

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
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    move_files(train_path, train_dir)
    move_files(val_path, val_dir)
