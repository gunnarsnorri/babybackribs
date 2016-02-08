#!/usr/bin/env python
import csv
import sys
import os
from shutil import copyfile

if len(sys.argv) > 2:
    source_dir = sys.argv[1].rstrip("/")
    target_dir = sys.argv[2].rstrip("/")
else:
    print("usage: %s source_dir target_dir" % __file__)
    sys.exit(1)

gt_file = "%s/gt.txt" % source_dir
with open(gt_file, 'r') as f:
    csv_reader = csv.reader(f, delimiter=";")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for row in csv_reader:
        image_file = row[0]
        category = row[5]
        if not os.path.exists("%s/%s" % (target_dir, category)):
            os.makedirs("%s/%s" % (target_dir, category))
        copyfile(
            "%s/%s" %
            (source_dir, image_file), "%s/%s/%s" %
            (target_dir, category, image_file))
