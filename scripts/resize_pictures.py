#!/usr/bin/env python
import os
import os.path
import argparse
from PIL import Image

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--indir", required=True,
                    help="Path to the source directory")
    ap.add_argument("-o", "--outdir", required=True,
                    help="Path to the output directory")
    ap.add_argument("-w", "--width", default=80, type=int,
                    help="Width of the frame")
    ap.add_argument("-he", "--height", default=80, type=int,
                    help="Width of the frame")
    args = vars(ap.parse_args())

    source_dir = args["indir"]
    target_dir = args["outdir"]
    width = args["width"]
    height = args["height"]
    filetypes = ["ppm", "jpg", "jpeg", "png", "bmp"]
    filenames = [filename for filename in os.listdir(source_dir) if
                 filename.split(".")[-1] in filetypes]
    for filename in filenames:
        img = Image.open(os.path.join(source_dir, filename))
        img = img.resize((width, height), 1)
        new_filename = filename.split(".")
        new_filename[0] = "%s-%dx%d" % (new_filename[0], width, height)
        new_filename = ".".join(new_filename)
        new_fullpath = os.path.join(target_dir, new_filename)
        img.save(new_fullpath)
