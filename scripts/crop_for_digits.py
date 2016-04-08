#!/usr/bin/env python

import cv2
import argparse
import os
import os.path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--txt", required=True, help="Path to annotation.txt")
    ap.add_argument("-i", "--imdir", required=True,
                    help="Path to image directory")
    ap.add_argument("-o", "--outdir", required=True,
                    help="Path to output directory")
    args = vars(ap.parse_args())

    txt_path = args["txt"]
    image_dir = args["imdir"]
    crop_dir = args["outdir"]
    last_image = None
    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)
    with open(txt_path) as read_f:
        for line in read_f:
            line = line.strip("\n")
            line = line.replace(" ","")
            image_name = line.split(";")[0]
            line = line.split(";")
            image_path = ('%s/%s' % (image_dir, image_name))
            image = cv2.imread(image_path)
            x = int(line[1])
            y = int(line[2])
            w = int(line[3])-int(line[1])
            h = int(line[4])-int(line[2])
            cropped_image = image[y:y+h, x:x+w]
            if image_name != last_image:
                i = 0
            crop_image_path = ('%s/%s/%s_%s.jpg' %
                               (crop_dir, line[5], image_name.split(".")[0], i))
            if not os.path.exists('%s/%s' % (crop_dir, line[5])):
                os.mkdir('%s/%s' % (crop_dir, line[5]))
            cv2.imwrite(crop_image_path, cropped_image)
            i += 1
            last_image = image_name
