#!/usr/bin/env python
import sys
import os
import os.path
from shutil import copyfile, rmtree
import pandas


def yn_choice(message, default='y'):
    choices = 'Y/n' if default.lower() in ('y', 'yes') else 'y/N'
    choice = raw_input("%s (%s) " % (message, choices))
    values = ('y', 'yes', '') if default == 'y' else ('y', 'yes')
    return choice.strip().lower() in values


def get_file_list(source_dir):
    gt_file_list = list(pandas.read_csv(
        os.path.join(source_dir, "gt.txt"), delimiter=";", header=None)[0])
    return [
        filename for filename in sorted(os.listdir(source_dir)) if
            filename in gt_file_list]

sets = [("train", 0.8), ("val", 0.9), ("test", 1.0)]

if len(sys.argv) > 2:
    source_dir = sys.argv[1].rstrip("/")
    target_dir = sys.argv[2].rstrip("/")
else:
    print("usage: %s source_dir target_dir" % __file__)
    sys.exit(1)

if os.path.exists(target_dir):
    msg = ("Warning: contents of %s will be removed, are you sure "
            "you want to continue?" % target_dir)
    if yn_choice(msg, default='n'):
        rmtree(target_dir)
    else:
        print("Aborted.")
        sys.exit(1)
for image_set, proportion in sets:
    os.makedirs(os.path.join(target_dir, image_set))


file_list = get_file_list(source_dir)
processed = 0.0
for filename in file_list:
    processed += 1.0/len(file_list)
    for image_set, proportion in sets:
        if processed < proportion:
            copyfile(
                os.path.join(source_dir, filename),
                os.path.join(target_dir, image_set, filename)
            )
            break
