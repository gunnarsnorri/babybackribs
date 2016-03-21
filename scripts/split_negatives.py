#!/usr/bin/env python
import os
import os.path
import argparse
import math
from PIL import Image

if __name__ == "__main__":
    """Splits input images into segments as defined by the
    height and width parameters"""
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
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    width = args["width"]
    height = args["height"]
    filetypes = ["ppm", "jpg", "jpeg", "png", "bmp"]
    filenames = [filename for filename in os.listdir(source_dir) if
                 filename.split(".")[-1] in filetypes]
    for filename in filenames:
        img = Image.open(os.path.join(source_dir, filename))
        size = img.size
        n = int(math.floor(size[0]/width))
        m = int(math.floor(size[1]/height))
        counter = 0
        try:
            leading_zeroes = int(math.ceil(math.log10(n*m)))
        except ValueError:
            print(
                "Skipping %s as crop size is greater than image size" %
                filename)
            continue
        format_string = "%%0%sd" % leading_zeroes
        for i in range(n):
            for j in range(m):
                partimg = img.crop(
                    (i*width, j*height, (i+1)*width, (j+1)*height))
                fname = filename.split(".")
                filenumber = format_string % counter
                fname[-2] = "%s_%s" % (fname[-2], filenumber)
                partfilename = ".".join(fname)
                partimg.save(os.path.join(target_dir, partfilename))
                counter += 1
