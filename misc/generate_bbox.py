#!/usr/bin/env python
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os
import dlib
import h5py
from skimage import io
import numpy as np

def run_dlib_selective_search(image_name):
    img = io.imread(image_name)
    rects = []
    dlib.find_candidate_object_locations(img,rects,min_size=0)
    proposals = []
    for k,d in enumerate(rects):
        #templist = [d.left(),d.top(),d.right(),d.bottom()]
        templist = [d.top(),d.left(),d.bottom(),d.right()]
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals


if __name__=="__main__":
    base_dir = "/mnt/nvme/traffic-rcnn/data/"
    imagenet_path = os.path.join(base_dir, 'Images')
    names = os.path.join(base_dir, "ImageSets/train.txt")
    
    count = 0
    imagenms = []
    trainfn = os.path.join(base_dir, "train.h5")
    with h5py.File(trainfn, "w") as trainFile:
        boxes_group = trainFile.create_group("boxes")
        with open(names) as nameFile: 
            for line in nameFile:
                imagename = line.rstrip('\n')
                imagenms.append(imagename)
                filename = os.path.join(imagenet_path,'%s.jpg' % imagename)
                proposals = run_dlib_selective_search(filename)
                boxes_group.create_dataset(imagename, data=proposals)
                count = count+1;
                print count
        
        trainFile.create_dataset("images", data=imagenms)
