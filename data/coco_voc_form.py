"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from .voc_eval import voc_eval
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


COCO_CLASSES = ( '__background__', # always index 0
                # 1-10
                'truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                'bench', 'elephant', 'bear', 'zebra', 'giraffe',
                # 11-20
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                # 21-30
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'wine glass', 'cup', 'fork', 'knife', 'spoon',
                # 31-40
                'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                # 41-50
                'cake', 'bed', 'toilet','laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                # 51-60
                'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class COCO_AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(COCO_CLASSES, range(len(COCO_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0,5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class COCODetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=None, n_support=1, n_query=10,
                 dataset_name='COCO60', phase='train'):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.split_index = list()
        self.n_support = n_support
        self.n_query = n_query
        self.phase = phase
        if phase == 'train' or phase == 'test_support':
            idx = 0
            for cls in COCO_CLASSES[1:]:
                self.split_index.append(idx)
                for (year, name) in image_sets:
                    for line in open(os.path.join(self.root, 'ImageSets', 'Main', cls + '_' + name + '_det.txt')):
                        self.ids.append((self.root, line.strip()))
                        idx += 1
            self.split_index.append(idx)
        elif phase == 'test_query':
            for (year, name) in image_sets:
                self._year = year
                for line in open(os.path.join(self.root, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append((self.root, line.strip()))
        # for (year, name) in image_sets:
        #     for line in open(os.path.join(self.root, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((self.root, line.strip()))
        # pass

    def __getitem__(self, index):

        indexes = self.split_index[index] + \
                 np.random.permutation(self.split_index[index+1]-self.split_index[index])[:(self.n_support + self.n_query)]
        img_ids = [self.ids[i] for i in indexes]
        imgs = []
        targets = []
        for img_id in img_ids:
            target = ET.parse(self._annopath % img_id).getroot()
            img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

            if self.target_transform is not None:
                target = self.target_transform(target)
                # ensure that groundtruth boxes with pos labels get priority during match
                idx = np.where(target[:, -1] == index + 1)
                tmp = target[idx]
                target = np.delete(target, idx, axis=0)
                target = np.concatenate((target, tmp), axis=0)
                if self.phase == 'train':
                    filter = target[:, -1] != index+1
                    target[filter, -1] = -1        # set the labels that are not belongs to this class to -1

            if self.preproc is not None:
                # img, target = self.preproc(img, target)
                img, target = self.preproc(img, target, index)
            imgs.append(img)
            targets.append(target)
        imgs = torch.stack(imgs, 0)
        support_imgs = imgs[:self.n_support]
        query_imgs = imgs[self.n_support:]
        support_targets = targets[:self.n_support]
        query_targets = targets[self.n_support:]

        if self.phase == 'test_support':
            return {'s_img': support_imgs, 's_target': support_targets}
        elif self.phase == 'train':
            return {'s_img': support_imgs, 's_target': support_targets, 'q_img': query_imgs, 'q_target': query_targets}
        else:
            print('The input phase is not applicable!')

    def __len__(self):
        if self.phase == 'train' or self.phase == 'test_support':
            return len(COCO_CLASSES) - 1
        elif self.phase == 'test_query':
            return len(self.ids)
        else:
            print('The input phase is not applicable!')

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def _get_voc_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(
            self.root, 'results', 'COCO' + self._year, 'Main') # "../data/COCO60/results/COCO2014/Main"
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(COCO_CLASSES):
            if cls == '__background__':
                continue
            print('Writing {} COCO results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls) # "../data/COCO60/results/COCO2014/Main"
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        name = self.image_set[0][1]
        annopath = os.path.join(
                                self.root,
                                'Annotations',
                                '{:s}.xml')
        imagesetfile = os.path.join(
                                self.root,
                                'ImageSets',
                                'Main',
                                name+'.txt')
        cachedir = os.path.join(self.root, 'annotations_cache') # "../data/COCO60/annotations_cache"
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True # if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(COCO_CLASSES):

            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                                    filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                                    use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
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

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    s_imgs = []
    s_targets = []
    q_imgs = []
    q_targets = []
    for _, sample in enumerate(batch):
        s_imgs.append(sample['s_img'])
        q_imgs.append(sample['q_img'])
        annos = [torch.from_numpy(i).float() for i in sample['s_target']]
        s_targets.append(annos)
        annos = [torch.from_numpy(i).float() for i in sample['q_target']]
        q_targets.append(annos)

    return (torch.stack(s_imgs, 0), s_targets, torch.stack(q_imgs, 0), q_targets)


class EpisodicBatchSampler(data.Sampler):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes  # 100

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way].tolist()