from __future__ import print_function
import os
import pickle
import argparse
from data import VOCroot
from data import VOC_AnnotationTransform, VOCDetection
from data.voc0712_meta import VOC_CLASSES
from data.coco_voc_form import COCO_CLASSES
from data.voc_eval import voc_eval
import numpy as np

parser = argparse.ArgumentParser(description='Receptive Field Block Net')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-r1', '--result1', type=str, help='Directory of result1 to compare')
parser.add_argument('-r2', '--result2', type=str, help='Directory of result2 to compare')
parser.add_argument('-c', '--cls_ind', default=1, type=int, help='')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
args = parser.parse_args()

def pr_curve_visualize():
    import matplotlib.pyplot as plt


    save_folder1 = os.path.join(args.save_folder, args.result1)
    save_folder2 = os.path.join(args.save_folder, args.result2)
    det_file1 = os.path.join(save_folder1, cls + '_pr.pkl')
    det_file2 = os.path.join(save_folder2, cls + '_pr.pkl')
    with open(det_file1,'rb') as f:
        result1 = pickle.load(f)
    with open(det_file2,'rb') as f:
        result2 = pickle.load(f)
    fig = plt.figure()
    fig.suptitle(cls)
    plt.plot(result1['rec'], result1['prec'], 'b', result2['rec'], result2['prec'], 'r')
    plt.show()
    pass

def do_python_eval(cls):
    rootpath = os.path.join(root, 'VOC' + year)
    name = 'test'
    annopath = os.path.join(
        rootpath,
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        rootpath,
        'ImageSets',
        'Main',
        name + '.txt')
    cachedir = os.path.join(root, 'annotations_cache')
    filename = get_voc_results_file_template().format(cls)
    rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=True)


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

    # load
    with open(cachefile, 'rb') as f:
        recs = pickle.load(f)

    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

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
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # number of detect boxes
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]] # a dict that consists of all groundtruth boxes that belong to this class in an image
        bb = BB[d, :].astype(float) # 一个
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float) # 多个

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
            ovmax = np.max(overlaps)
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

        BBGT = BBGT if BBGT.size == 0 else BBGT[jmax, :]
        det_visualize(classname, image_ids[d], sorted_scores[d], bb, BBGT, tp[d], fp[d], ovmax > ovthresh)

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return rec, prec

def det_visualize(cls, img_id, conf, bb, BBGT, tp, fp, is_ov):
    import cv2
    import matplotlib.pyplot as plt
    from data.voc0712_meta import VOC_CLASSES
    from data.coco_voc_form import COCO_CLASSES
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
    fig = plt.figure()
    BBGT = BBGT.astype(np.int16)
    bb = bb.astype(np.int16)
    if BBGT.size>0:
        cv2.rectangle(img, (BBGT[0], BBGT[1]), (BBGT[2], BBGT[3]), (0, 0, 1))  # ground truth
    if is_ov:

        if tp:
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 1, 0)) # tp
            fig.suptitle(cls + '_tp_'+ str(conf))
        elif fp: # already detected
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (1, 0, 0))  # fp
            fig.suptitle(cls + '_repeated_' + str(conf))
        else: # difficult
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 1, 0))  # fp
            fig.suptitle(cls + '_difficult_' + str(conf))
    else:
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (1, 0, 0))  # fp
        fig.suptitle(cls + '_not overlap_' + str(conf))
    plt.imshow(img)
    plt.show()

def get_voc_results_file_template():
    filename = 'comp4_det_test' + '_{:s}.txt'
    filedir = os.path.join(
        root, 'results', args.dataset + year, 'Main')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


if __name__ == '__main__':

    root = '../data/VOCdevkit' if args.dataset == 'VOC' else '../data/COCO60'
    year = '2007' if args.dataset == 'VOC' else '2014'
    cls = (VOC_CLASSES, COCO_CLASSES)[args.dataset == 'COCO'][args.cls_ind]
    # pr_curve_visualize()
    do_python_eval(cls)