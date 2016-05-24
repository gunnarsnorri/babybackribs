#!/usr/bin/env python

import os
import os.path
import argparse

def get_coords(_object):
    points = []
    for point in _object[1:5]:
        point = point.replace("l", "")
        points.append(int(round(float(point))))
    return tuple((points[i] for i in (2, 3, 0, 1)))



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--txt", required=True,
                    help="Path to the annotation txt")
    ap.add_argument("-o", "--outdir", required=True,
                    help="Path to the output directory")
    args = vars(ap.parse_args())
    txt_dir = args["txt"]
    target_dir = args["outdir"]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(txt_dir) as read_f:
        with open('%s/Annotations.txt' % target_dir, 'w') as write_f:
            for line in read_f:
                line = line.strip("\r\n")
                line = line.strip(" ")
                image_name = line.split(":")[0]
                if len(line.split(":")) > 1:
                    image_info = line.split(":")[1]
                    image_info = image_info.split(";")
                    del image_info[-1:]
                    objects = [x.split(", ") for x in image_info]
                    for _object in objects:
                        if len(_object) > 1:
                            points = get_coords(_object)
                            write_f.write(
                                '%s;%s;%s;%s;%s;%s\n' %
                                ((image_name,) + points + (_object[6],)))
