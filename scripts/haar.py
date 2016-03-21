import os
import cv2


def get_crops(xml, image_paths, crop_path):
    cascade = cv2.CascadeClassifier(xml)
    for image_path in image_paths:
        i = 0
        image = cv2.imread(image_path)
        # image,scaleFactor=1.1,minNeighbors =5
        # ,minSize=(10,10),flags(cv2.cv.CV_HAAR_SCALE_IMAGE
        signs = cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            maxSize=(200, 200),
            )
        for (x, y, w, h) in signs:
            image_name = image_path.split("/")[-1]
            crop_image_path = os.path.join(crop_path, "%s-%s" % (image_name, str(i)))
            cropped_image = image[y:y+h, x:x+w]
            cv2.imwrite(crop_image_path, cropped_image)
            i += 1
