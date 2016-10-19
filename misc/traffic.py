# ---------------------------------------
#  Fast R-CNN
#  Copyright (c) 2015 Microsoft
#  Licensed under the MIT License [see LICENSE for details]
#  Written by Ross Girshick
#  Edited by Jonas Tornqvist & Gunnar Snorri Ragnarsson
# -----------------------------------

import os
from datasets.imdb import imdb
# import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import h5py
import scipy.sparse
# import utils.cython_bbox
import cPickle
# import subprocess
import uuid
from traffic_eval import traffic_eval, parse_rec, get_max_overlap
# from fast_rcnn.config import cfg


class Traffic(imdb):

    def __init__(self, name, devkit_path):
        super(Traffic, self).__init__(name)
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__',  # always index 0
                         'sign')
        self._image_exts = ['.jpg', '.png', '.bmp', '.ppm']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': False,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}
        self.is_googlenet = False

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    @property
    def class_to_ind(self):
        return dict(zip(self.classes, xrange(self.num_classes)))

    def image_path_from_index(self, index):
        for ext in self._image_exts:
            image_path = os.path.join(self._data_path, 'Images',
                                      index + ext)
            if os.path.exists(image_path):
                return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self.name + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        pass

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # TODO: FIX cache annoyance
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_traffic_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self.name != "test":
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if self.name != "test":
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._data_path,
                                                self.name + '.h5'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        box_file = h5py.File(filename)
        raw_box_data = box_file["boxes"]

        box_list = []
        for image in raw_box_data:
            boxes = raw_box_data[image].value[
                :, (1, 0, 3, 2)]  # XXX: Check if correct
            # filter?
            box_list.append(boxes)

        # for i in xrange(raw_data.shape[0]):
        #     boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
        #     keep = ds_utils.unique_boxes(boxes)
        #     boxes = boxes[keep, :]
        #     keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
        #     boxes = boxes[keep, :]
        #     box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_traffic_annotation(self, index):

        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            class_name = obj.find('name').text
            cls = self.class_to_ind[class_name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2-x1)*(y2-y1)
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas
                }

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_results_file_template(self):
        filename = self._get_comp_id() + '_det_' + self.name + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_dataset_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = self._get_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        if self.is_googlenet:
                            det_gt_class = self.get_gt_class(dets[k, :-1], index)
                        else:
                            det_gt_class = ""
                        # TODO: check if dets[k, 0:4] should have +1
                        # or pascal-specific
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:s}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3], det_gt_class))

    def get_gt_class(self, det, image_index):
        """Gets corresponding ground truth to detections from annotations"""
        # Get annotations for image
        annopath = os.path.join(
            self._data_path, 'Annotations', '{:s}.xml'.format(image_index))
        recs = parse_rec(annopath)
        BBGT = np.array([x["bbox"] for x in recs]).astype(float)
        ovmax, jmax = get_max_overlap(BBGT, det)
        if ovmax > 0:
            return recs[jmax]["class"]
        else:
            return self._classes[0]

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._data_path, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(
            self._data_path, 'ImageSets', '%s.txt' % self.name)
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # TODO: Define ((0, 0, 0, 0),...) for all classes
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_results_file_template().format(cls)
            try:
                rec, prec, ap = traffic_eval(
                    filename,
                    annopath,
                    imagesetfile,
                    cls,
                    ovthresh=0.5,
                    googlenet=self.is_googlenet,
                )  # TODO: add ((tp,fp,tn,fn,....)
            except IOError:
                continue
            if ap is not None:
                aps.append([ap])
                print('{:.3f}'.format(ap))
            else:
                print("No AP for class %s" % cls)

        # TODO: Calculate skew for all classes
        # TODO: Calculate minAP from skew and table
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    # def _do_matlab_eval(self, output_dir='output'):
    #     print '-----------------------------------------------------'
    #     print 'Computing results with the official MATLAB eval code.'
    #     print '-----------------------------------------------------'
    #     path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
    #                         'VOCdevkit-matlab-wrapper')
    #     cmd = 'cd {} && '.format(path)
    #     cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    #     cmd += '-r "dbstop if error; '
    #     cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
    #            .format(self._devkit_path, self._get_comp_id(),
    #                    self.name, output_dir)
    #     print('Running:\n{}'.format(cmd))
    #     status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_dataset_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self.classes:
                if cls == '__background__':
                    continue
                filename = self._get_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = False

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2
            boxes[:, 2] = widths[i] - oldx1
            try:
                assert (boxes[:, 2] >= boxes[:, 0]).all()
            except AssertionError:
                import pdb
                pdb.set_trace()  # XXX BREAKPOINT
                raise
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'flipped': True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2


class TrafficMultiClass(Traffic):

    def __init__(self, name, devkit_path):
        super(TrafficMultiClass, self).__init__(name, devkit_path)
        self._classes = (
            '__background__',
            '30_SIGN',
            '50_SIGN',
            '60_SIGN',
            '70_SIGN',
            '80_SIGN',
            '90_SIGN',
            '100_SIGN',
            '110_SIGN',
            '120_SIGN',
            'GIVE_WAY',
            'NO_PARKING',
            'NO_STOPPING_NO_STANDING',
            'OTHER',
            'PASS_EITHER_SIDE',
            'PASS_LEFT_SIDE',
            'PASS_RIGHT_SIDE',
            'PEDESTRIAN_CROSSING',
            'PRIORITY_ROAD',
            'STOP',
            'URDBL'
        )


class GTSDBMultiClass(Traffic):

    def __init__(self, name, devkit_path):
        super(GTSDBMultiClass, self).__init__(name, devkit_path)
        self._classes = ('__background__',) + tuple([str(i) for i in range(43)])
