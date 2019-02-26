from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, VOC_AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc, EpisodicBatchSampler, COCO_AnnotationTransform
from layers.modules.multibox_loss_combined_meta_imprinted import MultiBoxLoss_combined
from layers.functions import PriorBox
import time
from data.voc0712 import VOC_CLASSES
from data.coco_voc_form import COCO_CLASSES
from logger import Logger
# torch.cuda.set_device(7)

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-w', '--n_way', default=20,
                    type=int, help='number of classes per episode (default: 20)')
parser.add_argument('--n_shot_task', type=int, default=1,
                    help="number of support examples per class on target domain(0 for whole dataset)")
parser.add_argument('--n_shot', type=int, default=1,
                    help="number of support examples per class during training (default: 1)")
parser.add_argument('--n_query', type=int, default=0,
                    help="number of query examples per class during training(default: 5)")
parser.add_argument('--train_episodes', type=int, default=70, # 500 for 3 shot, # 2521 for n_way = 12, 3782 for n_way = 8
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=2e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=40,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log', default=False,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    # train_sets = [('2014', 'train'), ('2014', 'valminusminival')]
    train_sets = [('2014', 'trainval')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    # from models.RFB_Net_vgg import build_net
    from models.RFB_Net_vgg_meta_imprinted import build_net
    # from models.RFB_Net_vgg_add_feature_layer import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unknown version!')

img_dim = (300, 512)[args.size == '512']
rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
p = (0.6, 0.2)[args.version == 'RFB_mobile']
# num_classes = (21, 61)[args.dataset == 'COCO']
num_classes = 61
overlap_threshold = 0.5
# weight_decay = 0.0005
# gamma = 0.1
# momentum = 0.9

net = build_net('train', img_dim, num_classes - 1, overlap_threshold)
print(net)
if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.obj.apply(weights_init)
    net.Norm.apply(weights_init)
    net.nonlinear.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
    # load resume network
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=False)

    for param in net.base.parameters():
        param.requires_grad = False
    for param in net.Norm.parameters():
        param.requires_grad = False
    for param in net.extras.parameters():
        param.requires_grad = False
    # for param in net.loc.parameters():
    #     param.requires_grad = False
    for param in net.conf.parameters():
        param.requires_grad = False
    # for param in net.obj.parameters():
    #     param.requires_grad = False
    # for param in net.nonlinear.parameters():
    #     param.requires_grad = False

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights for nonlinear mapping layers...')
    net.nonlinear.apply(weights_init)

optimizer = optim.SGD([
                            # {'params': net.base.parameters(), 'lr': args.lr*0.1},
                            # {'params': net.Norm.parameters(), 'lr': args.lr*0.5},
                            # {'params': net.extras.parameters(), 'lr': args.lr*0.5},
                            {'params': net.loc.parameters(), 'lr': args.lr*0.1},
                            {'params': net.obj.parameters(), 'lr': args.lr*0.1},
                            {'params': net.nonlinear.parameters()},
                            {'params': net.scale, 'lr': args.lr*0.1},
                        ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)
for group in optimizer.param_groups:
    group.setdefault('initial_lr', group['lr'])

# criterion = MultiBoxLoss(num_classes-1, 0.5, True, 0, True, 3, 0.5, False)
criterion = MultiBoxLoss_combined(num_classes - 1, overlap_threshold, True, 0, True, 3, 0.5, False, net)

if args.log:
    logger = Logger(args.save_folder + 'logs')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)), output_device=0)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()
num_priors = priors.size(0)

def train():
    net.train()
    epoch = 0 + args.resume_epoch

    print('Loading Dataset...')
    phase = 'train' if args.n_shot_task == 0 else 'meta_transfer'
    if args.dataset == 'VOC':
        # dataset = VOCDetection(VOCroot, train_sets, BaseTransform(
        #     img_dim, rgb_means, (2, 0, 1)), VOC_AnnotationTransform(), args.n_shot, args.n_query,
        #     phase=phase, n_shot_target=args.n_shot_target))
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), VOC_AnnotationTransform(), args.n_shot, args.n_query,
                               phase=phase, n_shot_task=args.n_shot_task)
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p), COCO_AnnotationTransform(), args.n_shot, args.n_query)
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = args.train_episodes
    max_iter = args.max_epoch * epoch_size

    # stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    # stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    # stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    milestones_VOC = [30, 35]
    milestones_COCO = [100, 150, 180]
    milestones = (milestones_VOC, milestones_COCO)[args.dataset == 'COCO']
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
    #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                            eps=1e-08)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.gamma, last_epoch=epoch - 1)

    print('Training', args.version, 'on', dataset.name)

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        t0 = time.time()
    else:
        start_iter = 0

    first_or_not = 1

    # sampler = EpisodicBatchSampler(n_classes=len(dataset), n_way=args.n_way,
    #                                n_episodes=args.train_episodes, phase='train')
    if args.n_shot_task == 1:
        sampler = EpisodicBatchSampler(n_classes=len(dataset), n_way=args.n_way,
                                       n_episodes=2*args.train_episodes, phase='test')
    else:
        sampler = EpisodicBatchSampler(n_classes=len(dataset), n_way=args.n_way,
                                       n_episodes=args.train_episodes, phase='train')
    # dataloader = iter(
    #     data.DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
    #                     collate_fn=detection_collate))
    # for i in range(10):
    #     s_img, s_t, q_img, q_t = next(dataloader)


    # # split the dataset according to classes
    # id_to_namedict = VOC_CLASSES
    # num = len(dataset)
    # cls_list_2007 = [[] for _ in range(21)]
    # cls_list_2012 = [[] for _ in range(21)]
    #
    # for index in range(len(dataset)):
    #     id, targets = dataset.pull_anno(index)
    #     nms = [] # 去除重复的标签
    #     for _, gt in enumerate(targets):
    #         if gt[-1] not in nms:
    #             nms.append(gt[-1])
    #             if len(id) == 6:
    #                 cls_list_2007[int(gt[-1])].append(id)
    #             else:
    #                 cls_list_2012[int(gt[-1])].append(id)
    #
    # for i in range(1, 21):
    #     with open('/home/zeyang/data/VOCdevkit/VOC2007/ImageSets/Main/%s_trainval_det.txt' % id_to_namedict[i], mode='w') as f:
    #         f.write('\n'.join(cls_list_2007[i]))
    #         f.write('\n')
    #     with open('/home/zeyang/data/VOCdevkit/VOC2012/ImageSets/Main/%s_trainval_det.txt' % id_to_namedict[i], mode='w') as f:
    #         f.write('\n'.join(cls_list_2012[i]))
    #         f.write('\n')
    #
    # id_to_namedict = COCO_CLASSES
    # num = len(dataset)
    # cls_list_COCO = [[] for _ in range(61)]
    #
    # for index in range(len(dataset)):
    #     id, targets = dataset.pull_anno(index)
    #     nms = [] # 去除重复的标签
    #     for _, gt in enumerate(targets):
    #         if gt[-1] not in nms:
    #             nms.append(gt[-1])
    #             cls_list_COCO[int(gt[-1])].append(id)
    #     print(str(index) + '/' + str(len(dataset)))
    #
    # for i in range(1, 61):
    #     with open('/home/zeyang/data/COCO60/ImageSets/Main/%s_trainval_det.txt' % id_to_namedict[i], mode='w') as f:
    #         f.write('\n'.join(cls_list_COCO[i]))
    #         f.write('\n')


    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            if args.n_shot_task == 1:
                batch_iterator = iter(data.DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                                                      collate_fn=lambda x: detection_collate(x, 'test')))
            else:
                batch_iterator = iter(data.DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                                                      collate_fn=lambda x: detection_collate(x, 'train')))


            if not first_or_not:
                print('Epoch' + repr(epoch) + ' Finished! || L: %.4f C: %.4f O: %.4f' % (
                    loc_loss / epoch_size, conf_loss / epoch_size, obj_loss / epoch_size)
                      )
                if epoch % 5 == 0 and epoch > 0:
                    if args.n_shot_task == 0 and args.dataset == 'COCO':
                        torch.save(net.state_dict(), args.save_folder + args.version + '_' + args.dataset + '60_meta_epoches_' +
                                   repr(epoch) + '.pth')
                    else:
                        torch.save(net.state_dict(),
                                   args.save_folder + args.version + '_' + args.dataset + '_meta_epoches_' +
                                   repr(epoch) + '.pth')
            loc_loss = 0
            conf_loss = 0
            obj_loss = 0

            epoch += 1
            scheduler.step()  # 等价于lr = args.lr * (gamma ** (step_index))
            lr = scheduler.get_lr()

        # if epoch < 6:  # warmup
        #     lr = adjust_learning_rate(optimizer, iteration, epoch_size)  # gamma = 0.1
        # for _ in range(4):
        # load train data
        if args.n_shot_task == 1:
            s_img, s_t = next(batch_iterator)
            q_img, q_t = next(batch_iterator)
            # for _ in range(4):
            #     tmp_img, tmp_t = next(batch_iterator)
            #     q_img = torch.cat((q_img, tmp_img), 1)
            #     q_t = [q_t[i]+tmp_t[i] for i in range(len(q_t))]
        else:
            s_img, s_t, q_img, q_t = next(batch_iterator)

        # index = torch.randperm(20)
        # s_img = torch.index_select(s_img, 0, index)
        # s_t = [s_t[id] for id in index]
        # q_img = torch.index_select(q_img, 0, index)
        # q_t = [q_t[id] for id in index]

        # img = torch.cat((s_img, q_img), 1)
        # tar = [s_t[i]+q_t[i] for i in range(len(s_t))]
        # vis_picture(img, tar)
        # vis_picture(s_img, s_t)
        # vis_picture(q_img, q_t)

        if args.cuda:
            s_img = Variable(s_img.cuda())
            q_img = Variable(q_img.cuda())
            s_t = [[Variable(anno.cuda()) for anno in cls_list] for cls_list in s_t]
            q_t = [[Variable(anno.cuda()) for anno in cls_list] for cls_list in q_t]
        else:
            s_img = Variable(s_img)
            q_img = Variable(q_img)
            s_t = [[Variable(anno) for anno in cls_list] for cls_list in s_t]
            q_t = [[Variable(anno) for anno in cls_list] for cls_list in q_t]

        # forward
        out = net((s_img, q_img))

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_obj = criterion(out, priors, (s_t, q_t))
        loss = loss_l + loss_c + loss_obj
        # loss_l, loss_c = criterion(out, priors, targets)
        # loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        obj_loss += loss_obj.item()

        if iteration % 10 == 0:
            if not first_or_not:
                t1 = time.time()
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                      + ' || Totel iter ' +
                      repr(iteration) + ' || L: %.4f C: %.4f O: %.4f ||' % (
                          loss_l.item(), loss_c.item(), loss_obj.item()) +
                      ' Time: %.4f sec. ||' % (t1 - t0) + ' LR: %.8f, %.8f' % (lr[0], lr[1]))
                if args.log:
                    logger.scalar_summary('loc_loss', loss_l.item(), iteration)
                    logger.scalar_summary('conf_loss', loss_c.item(), iteration)
                    logger.scalar_summary('obj_loss', loss_obj.item(), iteration)
                    logger.scalar_summary('lr', max(lr), iteration)
            t0 = time.time()

        first_or_not = 0
    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version + '_' + args.dataset + '_meta.pth')

# 单张图片可视化
# def vis_picture(im, targets):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import cv2
#     npimg = im.cpu().numpy()
#     # npimg = np.squeeze(npimg, 0)
#     im = np.transpose(npimg, (1, 2, 0))
#     im = (im + np.array([104, 117, 123])) / 255
#     im = im[:, :, ::-1].copy()
#
#     targets = targets.numpy()
#     labels = targets[:, -1]
#     boxes = targets[:, :4]
#     boxes = (boxes * 300).astype(np.uint16)
#     for i in range(boxes.shape[0]):
#         cv2.rectangle(im, (boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), (1, 0, 0))
#     cls = COCO_CLASSES[int(labels[0])]
#
#     plt.imshow(im)
#     plt.show()

def vis_picture(imgs, targets):
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

    for i in range(14, 15):
        CLASSES = (VOC_CLASSES, COCO_CLASSES)[args.dataset == 'COCO']
        cls = CLASSES[int(targets[i][0][-1, -1])]
        for j in range(per_way):
            fig = plt.figure()
            fig.suptitle(cls)
            # ax = fig.add_subplot(per_way, 1, j+1)
            img = imgs[i, j, :, :, :].copy()
            labels = targets[i][j][:, -1]
            boxes = targets[i][j][:, :4]
            boxes = (boxes * 300).astype(np.uint16)
            boxes_pos = boxes[labels != -1]
            boxes_neg = boxes[labels == -1]
            for k in range(boxes_neg.shape[0]):
                cv2.rectangle(img, (boxes_neg[k, 0], boxes_neg[k, 1]), (boxes_neg[k, 2], boxes_neg[k, 3]), (1, 0, 0))
            for k in range(boxes_pos.shape[0]):
                cv2.rectangle(img, (boxes_pos[k, 0], boxes_pos[k, 1]), (boxes_pos[k, 2], boxes_pos[k, 3]), (0, 1, 0))
            # cls = COCO_CLASSES[int(labels[0])]
            plt.imshow(img)
            plt.show()


def adjust_learning_rate(optimizer, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * 5)  # 前5个epoch有warm up的过程
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
