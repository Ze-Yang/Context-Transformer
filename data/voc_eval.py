# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):  # 第一次测的时候把annots从xml文件中提取出来,存在cache_file里
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # obj是这张图片里的物体,是一个dict
        bbox = np.array([x['bbox'] for x in R]) # 一张图片里某一类物体的所有gt
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R) # 未被detect过为False
        npos = npos + sum(~difficult) # 只检测非difficult的obj
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # txt文件存储格式image_ids confidence BB

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = confidence[sorted_ind]
    BB = BB[sorted_ind, :] if BB.size != 0 else BB
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # number of detect boxes
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # if image_ids[d] in ids:
        R = class_recs[image_ids[d]]  # a dict that consists of all groundtruth boxes that belong to this class in an image
        bb = BB[d, :].astype(float)  # 一个
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        '''
        # for visualization
        import cv2
        rootpath = os.path.join('../data/VOCdevkit/', 'VOC2007')
        imgpath = os.path.join(rootpath, 'JPEGImages', '%s.jpg')
        img = cv2.imread(imgpath % image_ids[d], cv2.IMREAD_COLOR)
        '''
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

                # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)    # 某一个bb框与BBGT框overlap的最大值
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

        # BBGT = BBGT if BBGT.size == 0 else BBGT[jmax, :]
        # det_visualize(classname, image_ids[d], sorted_scores[d], bb, BBGT, tp[d], fp[d], ovmax > ovthresh)
        # id = image_ids[d]
        # a = 0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def det_visualize(cls, img_id, conf, bb, BBGT, tp, fp, is_ov):
    '''
    blue for ground truth bounding box
    green for true positive detect result
    red for false positive detect result

    :param cls: class to visualize
    :param img_id: image id list
    :param conf: detect confidence
    :param bb: detect bounding box
    :param BBGT: ground truth bounding box
    :param tp: is true positive or not
    :param fp: is false positive or not
    :param is_ov: iou with ground truth bounding box surpass threshold or not
    :return:
    '''

    import cv2
    import matplotlib.pyplot as plt
    from .voc0712 import VOC_CLASSES
    from .coco_voc_form import COCO_CLASSES
    if cls in VOC_CLASSES:
        dataset = 'VOC'
    elif cls in COCO_CLASSES:
        dataset = 'COCO'
    else:
        print('The class is invalid.')
        return
    rootpath = '../data/VOCdevkit/VOC2007' if dataset == 'VOC' else '../data/COCO60'
    imgpath = os.path.join(rootpath, 'JPEGImages', '%s.jpg')
    img = cv2.imread(imgpath % img_id, cv2.IMREAD_COLOR)
    img = img[:, :, ::-1] / 255
    img = img.copy()
    # fig = plt.figure()
    BBGT = BBGT.astype(np.int16)
    bb = bb.astype(np.int16)
    if len(BBGT) > 0:
        for i in range(BBGT.shape[0]):
            cv2.rectangle(img, (BBGT[i, 0], BBGT[i, 1]), (BBGT[i, 2], BBGT[i, 3]), (0, 0, 1))  # ground truth
    if is_ov:

        if tp:
            pass
            fig = plt.figure()
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 1, 0)) # tp
            fig.suptitle(cls + '_tp_'+ str(conf))
            plt.imshow(img)
            plt.show()
        elif fp:  # already detected
            fig = plt.figure()
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (1, 0, 0))  # fp
            fig.suptitle(cls + '_repeated_' + str(conf))
            plt.imshow(img)
            plt.show()
        else: # difficult
            pass
            fig = plt.figure()
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 1, 0))  # fp
            fig.suptitle(cls + '_difficult_' + str(conf))
            plt.imshow(img)
            plt.show()
    else:
        if conf > 0.5:
            fig = plt.figure()
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (1, 0, 0))  # fp
            fig.suptitle(cls + '_not overlap_' + str(conf))
            plt.imshow(img)
            plt.show()
