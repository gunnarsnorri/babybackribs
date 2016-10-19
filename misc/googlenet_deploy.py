import numpy as np
import caffe
import cv2


class GoogLeNet(object):
    """Deployment wrapper for using googlenet to get proposals, given
    an image and associated pre-calculated bounding boxes"""

    def __init__(self, deploy_file, caffemodel, classes, mean, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', mean)
        self.transformer.set_transpose('data', (2, 0, 1))
        # self.transformer.set_raw_scale('data', 255.0)
        self.classes = classes

    def classify(self, image, confidence, boxes, threshold):
        """Classifies all boxes in one batch
        dets: [[pred, x1, y1, x2, y2],...]"""
        if type(image) == str:
            image = cv2.imread(image)
        inds = np.where(confidence >= threshold)[0]
        boxes = boxes[inds].astype(int)
        self.net.blobs['data'].reshape(boxes.shape[0], 3, 224, 224)
        crops = []
        # cv2.namedWindow("preview")
        for box in boxes:
            crop = image[box[1]:box[3]+1, box[0]:box[2]+1]
            crops.append(crop)
            # cv2.imshow("preview", crop)
            # cv2.waitKey(0)

        imgs = np.array([self.transformer.preprocess('data', i) for i in crops])
        # for img in imgs:
        #     cv2.imshow("preview", img.transpose((1,2,0)))
        #     cv2.waitKey(0)
        self.net.blobs['data'].data[...] = imgs
        return self.net.forward()


if __name__ == "__main__":
    labels_file = "/usr/share/digits/digits/jobs/20160530-164726-bf90/labels.txt"
    with open(labels_file, 'r') as lf:
        label_lines = lf.readlines()
    classes = [x.strip().replace(" ", "_") for x in label_lines]
    mean_file = "/mnt/nvme/caffe/models/bvlc_googlenet/mean.npy"
    mean = np.load(mean_file).mean(1).mean(1)
    g = GoogLeNet(
        "/mnt/nvme/caffe/models/bvlc_googlenet/deploy.prototxt",
        "/mnt/nvme/caffe/models/bvlc_googlenet/bvlc_googlenet_tune_iter_70000.caffemodel",
        classes,
        mean)
    detfile = "/mnt/nvme/traffic-dual-purpose/results/comp4_e65cab81-5100-48ad-9b7b-50f910730d17_det_test2_sign.txt"
    image_fullpath = "/mnt/nvme/traffic-dual-purpose/data/Images/1277104314Image000042.jpg"
    with open(detfile, 'r') as df:
        det_lines = df.readlines()
    splitlines = [x.strip().split(' ') for x in det_lines]
    confidence = np.array(
        [float(x[1]) for x in splitlines
         if x[0] == '1277104314Image000042'])
    bb = np.array(
        [[float(z) for z in x[2:6]] for x in splitlines
         if x[0] == '1277104314Image000042'])
    gt_classes = [x[6] for x in splitlines]
    classifications = g.classify(image_fullpath, confidence, bb, 0.0)
    classes = [g.classes[b.argmax()] for b in classifications["prob"]]
    print(gt_classes)
    print(classes)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
