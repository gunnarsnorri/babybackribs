import numpy as np
import cv2
from PIL import Image as edit

def get_crops(xml, image_paths, crop_path):
    cascade=cv2.CascadeClassifier(xml)
    for image_path in image_paths:
        int i = 0
        image = edit.open(image_path)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        signs = cascade.detectMultiscale(gray,1.1,5,0,(10,10),(200,200))
        for (x,y,w,h) in signs:
            crop_image_path=('%s/%s_%s')%(crop_path,i,image_path)
            cropped_image=edit.crop(image,x,y,w,h)
            cropped_image.save(crop_img_path)
            i+=1
