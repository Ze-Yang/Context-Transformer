from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from data import VOC_AnnotationTransform, COCO_AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512,\
    COCO_mobile_300, EpisodicBatchSampler, detection_collate, VOCroot, COCOroot, preproc
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
from utils.box_utils import match
from data.voc0712_meta import VOC_CLASSES
from data.coco_voc_form import COCO_CLASSES
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
# from sklearn import manifold

# np.random.seed(100)

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('--method', default='CAU',
                    help='CAU or TF(transfer)')
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
        from models.RFB_Net_vgg_imprinted import build_net
    elif args.method == 'TF':
        from models.RFB_Net_vgg import build_net
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
    img_dim = (300, 512)[args.size=='512']
    if args.method == 'CAU':
        feature_dim = 60
    elif args.method == 'TF':
        feature_dim = 20
    else:
        print('The value of args.method is not invalid.')
    net = build_net('test', img_dim, feature_dim)    # initialize detector
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
        dataset = VOCDetection(VOCroot, [('2007', 'test')], None,
                                VOC_AnnotationTransform(), phase='test_query')
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
