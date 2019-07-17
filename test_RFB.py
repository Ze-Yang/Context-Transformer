from __future__ import print_function
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from data import AnnotationTransform, COCO_AnnotationTransform, COCODetection, VOCDetection, BaseTransform, \
    VOC_300, VOC_512,COCO_300, COCO_512, COCO_mobile_300, VOCroot, COCOroot
from layers.functions import Detect, PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
# np.random.seed(100)

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('--method', default='CAU',
                    help='CAU, TF(transfer) or CAU_incre')
parser.add_argument('-m', '--trained_model', default='weights/RFB_vgg_VOC_epoches_190.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    if args.method == 'CAU':
        from models.RFB_Net_vgg_CAU import build_net
    elif args.method == 'TF':
        from models.RFB_Net_vgg import build_net
    elif args.method == 'CAU_incre':
        from models.RFB_Net_vgg_CAU_incre import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unkown version!')

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
num_classes = (21, 61)[args.dataset == 'COCO']
overlap_threshold = 0.5
num_priors = priors.size(0)
p = 0.6


def test_net(save_folder, net, detector, cuda, dataset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(dataset)

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)

        # class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        # id_to_index = dict(zip([i[1] for i in dataset.ids], range(len(dataset.ids))))
        #
        # cachedir = os.path.join('../data/VOCdevkit', 'annotations_cache')
        # cachefile = os.path.join(cachedir, 'annots.pkl')
        # imagesetfile = os.path.join(
        #                         '../data/VOCdevkit/VOC2007',
        #                         'ImageSets',
        #                         'Main',
        #                         'test.txt')
        # with open(cachefile, 'rb') as f:
        #     recs = pickle.load(f)
        # with open(imagesetfile, 'r') as f:
        #     lines = f.readlines()
        # imagenames = [x.strip() for x in lines]
        # class_recs = {}
        # for imagename in imagenames:
        #     R = [obj for obj in recs[imagename]] # obj是这张图片里的物体,是一个dict
        #     bbox = np.array([x['bbox'] for x in R]) # 一张图片里某一类物体的所有gt
        #     label = np.array([class_to_ind[x['name']] for x in R])
        #     difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        #     det = [False] * len(R)  # 未被detect过为False
        #     class_recs[imagename] = {'bbox': bbox,
        #                              'label': label,
        #                              'difficult': difficult,
        #                              'det': det}
        # imgs = os.listdir('/media/sde/zeyang/final/')
        # imgs = [m[:-4] for m in imgs]
        # # with open('/media/sde/zeyang/det.txt', 'r') as f:
        # #     lines = f.readlines()
        # # imgs = [x.strip() for x in lines]
        # for img_id, R in class_recs.items():
        #     # if not R['difficult'].any() and len(R['label']) == 1:
        #     if img_id in imgs:
        #         BBGT = R['bbox']
        #         label = R['label']
        #         BBGT = np.hstack((BBGT, label[:, np.newaxis]))
        #         det = np.zeros(BBGT.shape[0])
        #         bbs = []
        #         for i in range(1, len(VOC_CLASSES)):
        #             bb = all_boxes[i][id_to_index[img_id]]
        #             bb = np.hstack((bb, np.full((bb.shape[0], 1), i)))
        #             if len(bb) > 0:
        #                 bbs.append(bb)
        #         if len(bbs) > 0:
        #             bbs = np.concatenate(bbs, axis=0)
        #             sorted_ind = np.argsort(-bbs[:, -2])
        #             bbs = bbs[sorted_ind]
        #             result = np.full(bbs.shape[0], -1)  # -1 for not overlap or repeat det, 0 for wrong det, 1 for tp
        #
        #             for j in range(bbs.shape[0]):
        #                 # compute overlaps
        #                 # intersection
        #                 ixmin = np.maximum(BBGT[:, 0], bbs[j, 0])
        #                 iymin = np.maximum(BBGT[:, 1], bbs[j, 1])
        #                 ixmax = np.minimum(BBGT[:, 2], bbs[j, 2])
        #                 iymax = np.minimum(BBGT[:, 3], bbs[j, 3])
        #                 iw = np.maximum(ixmax - ixmin + 1., 0.)
        #                 ih = np.maximum(iymax - iymin + 1., 0.)
        #                 inters = iw * ih
        #
        #                 # union
        #                 uni = ((bbs[j, 2] - bbs[j, 0] + 1.) * (bbs[j, 3] - bbs[j, 1] + 1.) +
        #                        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
        #                        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #
        #                 overlaps = inters / uni
        #                 ovmax = np.max(overlaps)  # 某一个bb框与BBGT框overlap的最大值
        #                 jmax = np.argmax(overlaps)
        #                 if ovmax > 0.5:
        #                     if bbs[j, -1] == BBGT[jmax, -1]:
        #                         if det[jmax] == 0:
        #                             det[jmax] = 1
        #                             result[j] = 1
        #                     else:
        #                         if det[jmax] == 0:
        #                             det[jmax] = 2
        #                             result[j] = 0
        #             # if 1 in det or 0 in det:  # 漏检或误检
        #             # if 1 in det:
        #                 # idss.append(img_id)
        #             det_visualize(img_id, bbs, BBGT, result, det)
        # with open('/media/sde/zeyang/det.txt', 'w') as f:
        #     f.write('\n'.join(idss))
        #     f.write('\n')
        # # visualize predictions
        # for k in range(1, num_classes):
        #     for i in range(num_images):
        #         # if len(all_boxes[k][i]) > 0 and ((all_boxes[k][i][:, -1]>0.1) * (all_boxes[k][i][:, -1]<0.5)).any():
        #         if len(all_boxes[k][i]) > 0 and (all_boxes[k][i][:, -1] > 0.3).any():
        #             # boxes = np.empty([0, 5], dtype=np.float32)
        #             # if len(all_boxes[k][i]) > 0:
        #             #     boxes = np.row_stack((boxes, all_boxes[k][i]))
        #             # boxes = all_boxes[k][i][(all_boxes[k][i][:, -1] > 0.1) * (all_boxes[k][i][:, -1] < 0.5)]
        #             boxes = all_boxes[k][i][all_boxes[k][i][:, -1] > 0.3]
        #             img = test_query.pull_image(i)
        #             vis_picture_2(img, boxes, k)

        print('Evaluating detections')
        dataset.evaluate_detections(all_boxes, save_folder)
        return



    # mean_norm = torch.norm(s_conf_cls_mean_avg, dim=1)
    # norm_matrix = mean_norm.unsqueeze(1).mm(mean_norm.unsqueeze(0))
    # confusion_matrix = s_conf_cls_mean_avg.mm(s_conf_cls_mean_avg.t()) / norm_matrix
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(confusion_matrix.cpu().numpy())
    # plt.colorbar()
    # plt.show()

    # y = np.array(range(1, s_conf_cls_mean_avg.size(0)+1))
    # # t-SNE embedding of the digits dataset
    # print("Computing t-SNE embedding")
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time()
    # X_tsne = tsne.fit_transform(s_conf_cls_mean_avg)
    # plot_embedding(X_tsne, y,
    #                "t-SNE embedding of the digits (time %.2fs)" %
    #                (time() - t0))
    # plt.show()

    # # 清理中间变量
    # del truths
    # del labels
    # del s_img
    # del s_pos
    # del s_t
    # net = torch.nn.DataParallel(net, device_ids=list(range(1)), output_device=0)

    for i in range(num_images):
        img = dataset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()

        out = net(x)      # forward pass
        q_loc_data, q_conf_data, q_obj_data = out # q_conf_data[1, num_priors, feature_dim]
        q_conf = nn.functional.softmax(q_conf_data, dim=-1) # [1, num_priors, num_classes-1]
        q_obj_data = nn.functional.softmax(q_obj_data, dim=-1) # [1, num_priors, 2]
        pred = q_loc_data, q_conf, q_obj_data
        boxes, scores = detector.forward(pred, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0] # percent and point form detection boxes
        scores = scores[0] # [1, num_priors, 21]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

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
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    dataset.evaluate_detections(all_boxes, save_folder)


def det_visualize(img_id, bbs, BBGT, result, det):
    '''
    blue for ground truth bounding box
    green for true positive detect result
    red for false positive detect result
    :param img_id:
    :param bbs:
    :param BBGT:
    :param result:
    :param det:
    :return:
    '''
    colors = [(188, 143, 143), (0, 0, 255), (128, 42, 42),  # 'aeroplane', 'bicycle', 'bird'
              (255, 0, 255), (255, 255, 0), (0, 255, 0),  # 'boat', 'bottle', 'bus'
              (0, 255, 255), (8, 46, 84), (255, 0, 127),  # 'car', 'cat', 'chair'
              (128, 42, 42), (85, 102, 0), (34, 139, 34),  # 'cow', 'diningtable', 'dog'
              (255, 127, 0), (0, 127, 255), (127, 0, 255),  # 'horse', 'motorbike', 'person'
              (14, 255, 127), (64, 224, 205), (25, 25, 112),  # 'pottedplant', 'sheep', 'sofa'
              (255, 215, 0), (176, 100, 150)]  # 'train', 'tvmonitor'
    colors = [tuple([n / 255 for n in m]) for m in colors]
    cls_to_color = dict(zip(range(1, 21), colors))
    import cv2
    import matplotlib.pyplot as plt
    rootpath = '../data/VOCdevkit/VOC2007' if args.dataset == 'VOC' else '../data/COCO60'
    imgpath = os.path.join(rootpath, 'JPEGImages', '%s.jpg')
    img = cv2.imread(imgpath % img_id, cv2.IMREAD_COLOR)
    img = img[:, :, ::-1] / 255
    img = img.copy()
    fig = plt.figure()
    BBGT = BBGT.astype(np.int16)
    conf = bbs[:, -2].copy()
    col_idx = np.array([0, 1, 2, 3, 5])
    bbs = bbs[:, col_idx].astype(np.int16)
    bbs[bbs < 0] = 0
    # if len(BBGT) > 0:
    #     for i in range(BBGT.shape[0]):
    #         cv2.rectangle(img, (BBGT[i, 0], BBGT[i, 1]), (BBGT[i, 2], BBGT[i, 3]), (0, 0, 1))  # ground truth
    if len(bbs) > 0:
        for j in range(bbs.shape[0]):
            if result[j] == 1:
                cv2.rectangle(img, (bbs[j, 0], bbs[j, 1]), (bbs[j, 2], bbs[j, 3]), cls_to_color[bbs[j, -1]], 2)
            elif result[j] == 0:
                cls = VOC_CLASSES[bbs[j, -1]]
                cv2.rectangle(img, (bbs[j, 0], bbs[j, 1]), (bbs[j, 2], bbs[j, 3]), (1, 0, 0), 2)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    a = img_id
    pass


def vis_picture_1(imgs, targets):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    npimg = imgs.cpu().numpy()
    targets = [[anno.cpu().numpy() for anno in cls_list] for cls_list in targets]
    n_way = npimg.shape[0]
    per_way = npimg.shape[1]
    imgs = np.transpose(npimg, (0, 1, 3, 4, 2))
    imgs = (imgs + np.array([104, 117, 123])) / 255 # RGB
    imgs = imgs[:, :, :, :, ::-1] # BGR

    for i in range(14,15):
        CLASSES = (VOC_CLASSES, COCO_CLASSES)[n_way == 60]
        cls = CLASSES[int(targets[i][0][-1, -1])]
        for j in range(per_way):
            fig = plt.figure()
            fig.suptitle(cls)
            # ax = fig.add_subplot(per_way, 1, j+1)
            img = imgs[i, j, :, :, :].copy()
            labels = targets[i][j][:, -1]
            boxes = targets[i][j][:, :4]
            boxes = (boxes * 300).astype(np.uint16)
            for k in range(boxes.shape[0]):
                cv2.rectangle(img, (boxes[k, 0], boxes[k, 1]), (boxes[k, 2], boxes[k, 3]), (1, 0, 0))
            # cls = COCO_CLASSES[int(labels[0])]
            plt.imshow(img)
            plt.show()

def vis_picture_2(img, targets, cls_id):
    """
    Args:
        img: (numpy) Image to show
            Shape: [image_size, image_size, 3]
        targets: (numpy) bounding boxes
            Shape: each way is a list, each shot is a tensor, shape of the tensor[num_boxes, 5]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    img = img / 255 # RGB
    img = img[:, :, ::-1].copy() # BGR
    boxes = targets[:, :4]
    conf = min(targets[:, -1])
    boxes = boxes.astype(np.uint16)
    # boxes[:, 2] = np.clip(boxes[:, 2], 0, img.shape[1]-1)
    # boxes[:, 3] = np.clip(boxes[:, 3], 0, img.shape[0]-1)
    for k in range(boxes.shape[0]):
        cv2.rectangle(img, (boxes[k, 0], boxes[k, 1]), (boxes[k, 2], boxes[k, 3]), (0, 1, 0))
    CLASSES = (VOC_CLASSES, COCO_CLASSES)[n_way == 60]
    cls = CLASSES[cls_id]
    fig = plt.figure()
    fig.suptitle(cls + '(conf:' + str(conf) + ')')
    plt.imshow(img)
    plt.show()

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            # imagebox = offsetbox.AnnotationBbox(
            #     offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
            #     X[i])
            # ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


if __name__ == '__main__':
    # load net
    img_dim = (300, 512)[args.size == '512']
    if args.method == 'CAU':
        feature_dim = 60
    elif args.method == 'TF':
        feature_dim = 20
    elif args.method == 'CAU_incre':
        feature_dim = 15
    else:
        print('The value of args.method is not invalid.')
    net = build_net('test', img_dim, feature_dim)  # initialize detector
    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()
    print('Finished loading model!')
    print(net)
    # load data
    if args.dataset == 'VOC':
        if args.method == 'CAU_incre':
            from data.voc0712_incre import VOCDetection, AnnotationTransform
            dataset = VOCDetection(VOCroot, [('2007', 'test')], None,
                                   AnnotationTransform(), phase='test')
        else:
            dataset = VOCDetection(VOCroot, [('2007', 'test')], None,
                                   AnnotationTransform(), phase='test')
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, [('2014', 'test')], None,
                                  COCO_AnnotationTransform(), phase='test_query')
    else:
        print('Only VOC and COCO are supported now!')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    # evaluation
    #top_k = (300, 200)[args.dataset == 'COCO']
    top_k = 200
    detector = Detect(num_classes, 0, cfg)
    # save_folder = os.path.join(args.save_folder, args.dataset)
    test_net(args.save_folder, net, detector, args.cuda, dataset,
             BaseTransform(net.size, rgb_means, (2, 0, 1)),
             top_k, thresh=0.01)
