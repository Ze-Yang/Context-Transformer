from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_SSD_300, COCO_SSD_300, BaseTransform, preproc, EpisodicBatchSampler
from data.voc0712 import AnnotationTransform, VOCDetection, detection_collate
from layers.modules.multibox_loss_combined import MultiBoxLoss_combined
from layers.functions import PriorBox
import time
from logger import Logger

parser = argparse.ArgumentParser(
    description='SSD Training')
parser.add_argument('-v', '--version', default='SSD_vgg',
                    help='SSD_vgg ,SSD_E_vgg or SSD_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=64,
                    type=int, help='Batch size for training')
parser.add_argument('--n_shot_task', type=int, default=5,
                    help="number of support examples per class on target domain")
parser.add_argument('--train_episodes', type=int, default=100,
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float, help='initial learning rate')
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
    cfg = VOC_SSD_300
else:
    # train_sets = [('2014', 'train'), ('2014', 'valminusminival')]
    train_sets = [('2014', 'trainval')]
    cfg = COCO_SSD_300

if args.version == 'SSD_vgg':
    from models.SSD_Net_vgg import build_net
else:
    print('Unknown version!')

img_dim = (300, 512)[args.size == '512']
rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'SSD_mobile']
p = (0.6, 0.2)[args.version == 'SSD_mobile']
num_classes = (21, 61)[args.dataset == 'COCO']
overlap_threshold = 0.5

net = build_net('train', img_dim, num_classes - 1)
print(net)
if args.resume_net is None:
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
    net.Norm.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.obj.apply(weights_init)
    if args.version == 'SSD_E_vgg':
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
        if 'vgg' in k:
            name = 'base' + k[3:]
        elif 'L2Norm' in k:
            name = 'Norm' + k[6:]
        else:
            name = k
        if 'conf' in name: # do not load parameters for the last layer
            pass
        else:
            new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=False)
    # for param in net.base.parameters():
    #     param.requires_grad = False
    # for param in net.Norm.parameters():
    #     param.requires_grad = False
    # for param in net.extras.parameters():
    #     param.requires_grad = False
    # for param in net.conf.parameters():
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

    print('Initializing weights for the classifier...')
    net.conf.apply(weights_init)

optimizer = optim.SGD([
                            {'params': net.base.parameters(), 'lr': args.lr},
                            {'params': net.Norm.parameters(), 'lr': args.lr},
                            {'params': net.extras.parameters(), 'lr': args.lr},
                            {'params': net.loc.parameters()},
                            {'params': net.conf.parameters()},
                            {'params': net.obj.parameters()}
                        ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)
for group in optimizer.param_groups:
    group.setdefault('initial_lr', group['lr'])

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)), output_device=0)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

# criterion = MultiBoxLoss(num_classes-1, 0.5, True, 0, True, 3, 0.5, False)
criterion = MultiBoxLoss_combined(num_classes - 1, overlap_threshold, True, 0, True, 3, 0.5, False)

if args.log:
    logger = Logger(args.save_folder + 'logs')

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
    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform(), n_shot_task=args.n_shot_task)
    else:
        print('Only VOC is supported now!')
        return

    epoch_size = args.train_episodes
    max_iter = args.max_epoch * epoch_size

    milestones_VOC = [30, 35]
    milestones_COCO = [30, 60, 90]
    milestones = (milestones_VOC, milestones_COCO)[args.dataset == 'COCO']

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.gamma, last_epoch=epoch - 1)

    print('Training', args.version, 'on', dataset.name)

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        t0 = time.time()
    else:
        start_iter = 0

    first_or_not = 1

    sampler = EpisodicBatchSampler(n_classes=len(dataset), n_way=args.batch_size,
                                   n_episodes=args.train_episodes, phase='train')

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            if not first_or_not:
                print('Epoch' + repr(epoch) + ' Finished! || L: %.4f C: %.4f O: %.4f' % (
                    loc_loss / epoch_size, conf_loss / epoch_size, obj_loss / epoch_size)
                      )
                if epoch % 5 == 0:
                    torch.save(net.state_dict(), args.save_folder + args.version + '_' + args.dataset + '_tf_epoches_' +
                               repr(epoch) + '.pth')
            loc_loss = 0
            conf_loss = 0
            obj_loss = 0

            epoch += 1
            scheduler.step()  # 等价于lr = args.lr * (gamma ** (step_index))
            lr = scheduler.get_lr()

        # load train data
        images, targets = next(batch_iterator)  # [n_way, n_shot, 3, im_size, im_size]

        # vis_picture(images, targets)

        if args.cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        else:
            targets = [anno for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_obj = criterion(out, priors, targets)
        loss = loss_l + loss_c + loss_obj
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
                      ' Time: %.4f sec. ||' % (t1 - t0) + ' LR: %.8f, %.8f' % (lr[0], lr[3]))
                if args.log:
                    logger.scalar_summary('loc_loss', loss_l.item(), iteration)
                    logger.scalar_summary('conf_loss', loss_c.item(), iteration)
                    logger.scalar_summary('obj_loss', loss_obj.item(), iteration)
                    logger.scalar_summary('lr', max(lr), iteration)
            t0 = time.time()

        first_or_not = 0
    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version + '_' + args.dataset + '_tf.pth')
    # torch.save(net.state_dict(), args.save_folder +
    #            'Final_' + args.version + '_' + args.dataset + '_meta_tf.pth')


def vis_picture(imgs, targets):
    """
    Args:
        imgs: (tensor) Image to show
            Shape: [n_way, n_shot, 3, image_size, image_size]
        targets: (list) bounding boxes
            Shape: each way is a list, each shot is a tensor, shape of the tensor[num_boxes, 5]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    np_img = imgs.cpu().numpy()
    targets = [anno.cpu().numpy() for anno in targets]
    num = imgs.shape[0]
    imgs = np.transpose(np_img, (0, 2, 3, 1))
    imgs = (imgs + np.array([104, 117, 123])) / 255 # RGB
    imgs = imgs[:, :, :, ::-1] # BGR

    for i in range(num):
        img = imgs[i, :, :, :].copy()
        labels = targets[i][:, -1]
        boxes = targets[i][:, :4]
        boxes = (boxes * 300).astype(np.uint16)
        for k in range(boxes.shape[0]):
            cv2.rectangle(img, (boxes[k, 0], boxes[k, 1]), (boxes[k, 2], boxes[k, 3]), (0, 1, 0))
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