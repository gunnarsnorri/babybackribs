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
from voc_eval import voc_eval
# from fast_rcnn.config import cfg


class Traffic(imdb):

    def __init__(self, image_set, devkit_path=None):
        super(Traffic, self).__init__(image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__',  # always index 0
                         'sign')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_exts = ['.jpg', '.png', '.bmp']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

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
                                      self._image_set + '.txt')
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

        if self._image_set != "test":
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
        if self._image_set != "test":
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
            # Dont know what dict, zip and xrange
            cls = self._class_to_ind['sign']
            # does in the original call in _class_to_ind
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
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._data_path,
            'results',
            filename)
        return path

    def _write_dataset_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        # Have to track this, i have no idea why this is set in stone.
        comp_id = 'comp4'
        if use_salt:
            comp_id += '_{}'.format(os.getpid())  # diff between
            # - and _ in inria and original?
        path = os.path.join(
            self._devkit_path,
            'results',
            self.name,
            comp_id + '_')

        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue  # assumes continue skips this iteration of the loop.
            print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._data_path, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(
            self._data_path, 'ImageSets', 'train.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_results_file_template().format(cls)
            try:
                rec, prec, ap = voc_eval(
                    filename,
                    annopath,
                    imagesetfile,
                    cls,
                    cachedir,
                    ovthresh=0.5,
                    use_07_metric=False)
            except IOError:
                continue
            aps += [ap]

            print('{:.3f}'.format(ap))
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
    #                    self._image_set, output_dir)
    #     print('Running:\n{}'.format(cmd))
    #     status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
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
            self.config['cleanup'] = True
