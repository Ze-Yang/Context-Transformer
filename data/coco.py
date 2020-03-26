"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import pickle
import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
import itertools

from utils.pycocotools.coco import COCO
from utils.pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from collections import OrderedDict


_PREDEFINED_SPLITS_COCO = {
    "train2014": ("train2014", "annotations/instances_train2014.json"),
    "val2014": ("val2014", "annotations/instances_val2014.json"),
    "minival2014": ("val2014", "annotations/instances_minival2014.json"),
    "valminusminival2014": (
        "val2014",
        "annotations/instances_valminusminival2014.json",
    ),
    "split_nonvoc_train2014": (
        "train2014",
        "annotations/split_nonvoc_instances_train2014.json",
    ),
    "split_voc_train2014": (
        "train2014",
        "annotations/split_voc_instances_train2014.json",
    ),
    "split_nonvoc_val2014": (
        "val2014",
        "annotations/split_nonvoc_instances_val2014.json",
    ),
    "split_voc_val2014": (
        "val2014",
        "annotations/split_voc_instances_val2014.json",
    ),
    "split_nonvoc_minival2014": (
        "val2014",
        "annotations/split_nonvoc_instances_minival2014.json",
    ),
    "split_voc_minival2014": (
        "val2014",
        "annotations/split_voc_instances_minival2014.json",
    ),
    "split_nonvoc_valminusminival2014": (
        "val2014",
        "annotations/split_nonvoc_instances_valminusminival2014.json",
    ),

    "split_voc_valminusminival_2014": (
        "val2014",
        "annotations/split_voc_instances_valminusminival2014.json",
    ),

}


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

    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='COCO'):
        self.root = root
        self.cache_path = os.path.join(self.root, 'cache')
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self.annotations = list()

        for (year, image_set) in image_sets:
            coco_name = image_set+year
            image_root = os.path.join(root, 'images', _PREDEFINED_SPLITS_COCO[coco_name][0])
            annofile = os.path.join(root, _PREDEFINED_SPLITS_COCO[coco_name][1])
            self._COCO = COCO(annofile)
            self.coco_name = coco_name
            self.class_name = self._get_coco_instances_meta()
            self.num_classes = len(self.class_name)
            self.img_ids = sorted(self._COCO.imgs.keys())
            imgs = self._COCO.loadImgs(self.img_ids)
            self.ids.extend([os.path.join(image_root, img["file_name"]) for img in imgs])
            self.annotations.extend(self._load_coco_annotations(coco_name, self.img_ids, self._COCO))

    def _load_coco_annotations(self, coco_name, indexes, _COCO):
        cache_file=os.path.join(self.cache_path, coco_name+'_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(coco_name, cache_file))
            return roidb

        gt_roidb = [self._annotation_from_index(index, _COCO)
                    for index in indexes]
        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _get_coco_instances_meta(self):
        thing_ids = self._COCO.getCatIds()
        cats = self._COCO.loadCats(thing_ids)
        cats_name = [c['name'] for c in cats]
        self._class_to_coco_cat_id = dict(zip(cats_name, thing_ids))

        voc_inds = (0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62)
        nonvoc_inds = tuple([i for i in range(80) if i not in voc_inds])
        if 'nonvoc' in self.coco_name:
            self.id_map = nonvoc_inds
            thing_ids = [thing_ids[i] for i in self.id_map]
            thing_classes = [cats_name[k] for k in self.id_map]
        elif 'voc' in self.coco_name:
            self.id_map = voc_inds
            thing_ids = [thing_ids[i] for i in self.id_map]
            thing_classes = [cats_name[k] for k in self.id_map]
        self._thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids, 1)}
        return thing_classes

    def _annotation_from_index(self, index, _COCO):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = _COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        # Lookup table to map from COCO category ids to our internal class
        # indices
        for ix, obj in enumerate(objs):
            cls = self._thing_dataset_id_to_contiguous_id[obj['category_id']]
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = cls

        return res

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = self.annotations[index]
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        # in order to be compatible with mixup
        weight = np.ones((target.shape[0], 1))
        target = np.hstack((target, weight))

        return img, target

    def __len__(self):
        return len(self.ids)

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
        return cv2.imread(img_id, cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def _do_detection_eval(self, res_file):
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        results = OrderedDict()
        results['bbox'] = self._derive_coco_results(coco_eval, 'bbox', class_names=self.class_name)
        print_csv_format(results)

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.img_ids):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
              [{'image_id': index,
                'category_id': cat_id,
                'bbox': [xs[k], ys[k], ws[k], hs[k]],
                'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.class_name, 1):
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)
            fid.flush()

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, 'detections_' + self.coco_name + '_results')
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            self._do_detection_eval(res_file)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            print("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
        print(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        # assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in zip(self.id_map, class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        print("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    assert isinstance(results, OrderedDict), results  # unordered results cannot be properly printed
    for task, res in results.items():
        # Don't print "AP-category" metrics since they are usually not tracked.
        important_res = [(k, v) for k, v in res.items() if "-" not in k]
        print("copypaste: Task: {}".format(task))
        print("copypaste: " + ",".join([k[0] for k in important_res]))
        print("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
