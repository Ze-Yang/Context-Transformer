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
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss, MultiBoxLoss_combined
from layers.functions import PriorBox
import time
from data.voc0712 import VOC_CLASSES

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
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
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
    train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
    # from models.RFB_Net_vgg_add_feature_layer import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unknown version!')

img_dim = (300,512)[args.size=='512']
rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
p = (0.6,0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
# weight_decay = 0.0005
# gamma = 0.1
# momentum = 0.9

net = build_net('train', img_dim, num_classes-1)
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
    net.Norm.apply(weights_init)
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
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
#optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)
for group in optimizer.param_groups:
    group.setdefault('initial_lr', group['lr'])

# criterion = MultiBoxLoss(num_classes-1, 0.5, True, 0, True, 3, 0.5, False)
criterion = MultiBoxLoss_combined(num_classes-1, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()



def train():
    net.train()
    epoch = 0 + args.resume_epoch

    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    # stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    # stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    # stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    milestones_VOC = [150, 200, 250]
    milestones_COCO = [90, 120, 140]
    milestones = (milestones_VOC, milestones_COCO)[args.dataset == 'COCO']
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
    #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                            eps=1e-08)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.gamma, last_epoch=epoch-1)

    print('Training',args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        t0 = time.time()
    else:
        start_iter = 0

    first_or_not = 1

    '''
    split the dataset according to classes
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
    '''

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))

            if not first_or_not:
                print('Epoch' + repr(epoch) + ' Finished! || L: %.4f C: %.4f O: %.4f' % (
                          loc_loss/epoch_size, conf_loss/epoch_size, obj_loss/epoch_size)
                          )
                if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                    torch.save(net.state_dict(), args.save_folder + args.version + '_' + args.dataset + '_epoches_' +
                               repr(epoch) + '.pth')
            loc_loss = 0
            conf_loss = 0
            obj_loss = 0

            epoch += 1
            scheduler.step()  # 等价于lr = args.lr * (gamma ** (step_index))
            lr = scheduler.get_lr()[0]

        # if iteration in stepvalues:
        #     step_index += 1   # 在resume模式下有bug
        # lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size) # gamma = 0.1
        if epoch < 6: # warmup
            lr = adjust_learning_rate(optimizer, iteration, epoch_size)  # gamma = 0.1


        # load train data
        images, targets = next(batch_iterator)
        
        #print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]
        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_obj = criterion(out, priors, targets)
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
                        ' Time: %.4f sec. ||' % (t1 - t0) + ' LR: %.8f' % (lr))
                # print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                #       + ' || Totel iter ' +
                #       repr(iteration) + ' || L: %.4f C: %.4f ||' % (
                #           loss_l.item(), loss_c.item()) +
                #       ' Time: %.4f sec. ||' % (t1 - t0) + ' LR: %.8f' % (lr))
            t0 = time.time()

        first_or_not = 0
    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version +'_' + args.dataset+ '.pth')


def adjust_learning_rate(optimizer, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) # 前5个epoch有warm up的过程
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
