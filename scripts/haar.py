import numpy as np
import cv2


def get_crops(xml, image_paths, crop_path):
    cascade=cv2.CascadeClassifier(xml)
    for image_path in image_paths:
        i = 0
        image = cv2.imread(image_path)
        signs = cascade.detectMultiscale(image,scaleFactor=1.1,minNeighbors =5 ,minSize=(10,10),flags(cv2.cv.CV_HAAR_SCALE_IMAGE)
        for (x,y,w,h) in signs:
            crop_image_path=('%s/%s_%s')%(crop_path,i,image_path)
            cropped_image=image[x:x+w,y:y+h]
            cv2.imwrite(crop_img_path, cropped_image)
            i+=1
