from __future__ import print_function
import os
import pickle
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, \
    VOC_300, VOC_512, COCO_300, COCO_512, VOCroot, COCOroot
from layers.functions import Detect, PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
from utils.logger import setup_logger
from utils.checkpointer import DetectionCheckpointer

parser = argparse.ArgumentParser(description='Context-Transformer')
# Model and Dataset
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('--load-file', default=None,
                    help='Model checkpoint for loading.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version.')
parser.add_argument('--split', type=int, default=1,
                    help='VOC base/novel split, for VOC only.')
# Testing Parameters
parser.add_argument('--setting', default='transfer',
                    help='Testing setting: transfer or incre.')
parser.add_argument('-p', '--phase', type=int, default=1,
                    help='Testing phase. 1: source pretraining, 2: target fintuning.')
parser.add_argument('--method', default='ours',
                    help='ft(baseline) or ours, for phase 2 only.')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Use cuda to train model.')
parser.add_argument('--cpu', type=bool, default=False,
                    help='Use cpu nms.')
parser.add_argument('--retest', action='store_true',
                    help='Test cache results.')
parser.add_argument('--resume', action='store_true',
                    help='Whether to test the last checkpoint.')
parser.add_argument('--save-folder', default='weights/', type=str,
                    help='Dir to save results.')
args = parser.parse_args()

if args.dataset == 'VOC':
    test_set = [('2007', 'test')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
elif args.dataset == 'COCO':
    test_set = [('2014', 'split_nonvoc_minival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

if args.phase == 1:
    from models.RFB_Net_vgg import build_net
    if args.dataset == 'VOC':
        src_cls_dim = 15
        num_classes = 16  # include background
    else:
        src_cls_dim = 60
        num_classes = 61  # include background
elif args.phase == 2:
    if args.setting == 'transfer':
        if args.method == 'ours':
            from models.RFB_Net_vgg import build_net
            src_cls_dim = 60
            num_classes = 21
        elif args.method == 'ft':
            from models.RFB_Net_vgg import build_net
            src_cls_dim = 20
            num_classes = 21
        else:
            raise ValueError(f"Unknown method: {args.method}")
    elif args.setting == 'incre':
        if args.method == 'ours':
            from models.RFB_Net_vgg import build_net
            src_cls_dim = 15
            num_classes = 21
        else:
            raise ValueError('We only support our method for incremental setting.')
    else:
        raise ValueError(f"Unknown setting: {args.setting}")
else:
    raise ValueError(f"Unknown phase: {args.phase}")

img_dim = (300, 512)[args.size == '512']
rgb_means = (104, 117, 123)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def do_test(args, model, detector, max_per_image=200, thresh=0.01):
    if args.dataset == 'VOC':
        dataset = VOCDetection(args, VOCroot, [('2007', 'test')], None,
                               AnnotationTransform(0 if args.setting == 'transfer' else args.split), True)
    elif args.dataset == 'COCO':
        dataset = COCODetection(
            COCOroot, [('2014', 'split_nonvoc_minival')], None)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    transform = BaseTransform(model.size, rgb_means, (2, 0, 1))

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(args.save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        logger.info('Evaluating detections')
        dataset.evaluate_detections(all_boxes, args.save_folder)
        return

    for i in range(num_images):
        img = dataset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]]).to(model.device)
        with torch.no_grad():
            x = transform(img).unsqueeze(0)

        _t['im_detect'].tic()

        pred = model(x)  # forward pass
        boxes, scores = detector.forward(pred, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]  # percent and point form detection boxes
        scores = scores[0]  # [1, num_priors, num_classes]

        boxes *= scale  # scale each detection back up to the image
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            logger.info('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Evaluating detections')
    dataset.evaluate_detections(all_boxes, args.save_folder)


if __name__ == '__main__':
    logger = setup_logger(os.path.join(args.save_folder, 'inference'))
    # load net
    model = build_net(args, img_dim, src_cls_dim).eval()
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, args).resume_or_load(
        args.load_file, resume=args.resume
    )

    args.save_folder = os.path.join(args.save_folder, 'inference')
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.cuda and torch.cuda.is_available():
        model.device = 'cuda'
        model.cuda()
        cudnn.benchmark = True
    else:
        model.device = 'cpu'

    detector = Detect(num_classes, 0, cfg)
    do_test(args, model, detector)
