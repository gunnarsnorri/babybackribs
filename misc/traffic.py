#---------------------------------------
# Fast R-CNN
# For traffic signs
# written by Jonas Tornqvist
#-----------------------------------

import cPickle
import os 
import datasets.imdb
import scipy.sparse


class traffic(datasets.imdb):
    def __init__(self, image_set, devkit_path=None)
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path =  devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes= ('__background', #always index 0
                        'sign')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg','.png','.bmp']
        self._image_index = self._load_image_set_index()
        # default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        
        # Specific config options
        self.config={'cleanup'   : True,
                     'use_salt'  : True,
                     'top_k'     : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        '''
        see master comment (smc)
        '''
        return self.image_path_from_index(self._image_index[i])
    def image_path_from_index(self, index)
        #smc
        for ext in self._image_ext: #for every extension defined in self
            image_path = os.path.join(self._data_path, 'Images',
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set, '.txt')
        assert os.path.exists(image_set_file), \
                'path does not exist: {}'.format(imageset_file)
        with open(image_set_file) as f:
            image_index = x.strip() for x in f.readlines()]
        return image_index
   
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

        gt_roidb = [self._load_dataset_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_dataset_annotation(self, index):

        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')

        with open(filename) as f:
            data = f.read()
        # handle the data from the txt file
        # save all lines containing data in objs
        for ix, obj in enumerate(objs)
            x1 =
            y1 =
            x2 =
            y2 =
            cls = self._class_to_ind['sign'] #Dont know what dict, zip and xrange
                                             #does in the original call in _class_to_ind
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix,cls] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)
    
        return {'boxes'       : boxes,
                'gt_classes'  : gt_classes,
                'gt_overlaps' : overlaps,
                'flipped'     : False
               }
      
    def _write_dataset_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4' #Have to track this, i have no idea why this is set in stone.
        if use_salt:
            comp_id += '_{}'.format(os.getpid()) #diff between
                                                 # - and _ in inria and original? 
        path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue #assumes continue skips this iteration of the loop.
            print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind]im_ind[]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))}
        return comp_id
    def _do_python_eval(self, output_dir = 'output')
        #ska skrivas

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_dataset_results_file(all_boxes)
        self._do_python_eval(output_dir)
        
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d =  datasets.dataset('train', '')
    rest = d.roidb
    from IPython import embed; embed()















