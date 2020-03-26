from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from torch.nn.parallel import DataParallel, DistributedDataParallel
from data import AnnotationTransform, VOCDetection, COCODetection, detection_collate, VOCroot, COCOroot, \
    VOC_300, VOC_512, COCO_300, COCO_512, preproc
from models.RFB_Net_vgg import build_net
from layers.modules.multibox_loss_combined import MultiBoxLoss_combined
from layers.functions import PriorBox
from utils.box_utils import match
from utils.solver import build_optimizer, build_lr_scheduler
from utils.checkpointer import DetectionCheckpointer, PeriodicCheckpointer
from utils.logger import setup_logger
from utils.sampler import TrainingSampler
from utils.event import EventStorage, CommonMetricPrinter, TensorboardXWriter
# np.random.seed(100)

parser = argparse.ArgumentParser(
    description='Context-Transformer')

# Model and Dataset
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset.')
parser.add_argument('--split', type=int, default=1,
                    help='VOC base/novel split, for VOC only.')

# Training Parameters
parser.add_argument('--setting', default='transfer',
                    help='Training setting: transfer or incre.')
parser.add_argument('-p', '--phase', type=int, default=1,
                    help='Training phase. 1: source pretraining, 2: target fintuning.')
parser.add_argument('-m', '--method', default='ours',
                    help='ft(baseline) or ours, for phase 2 only.')
parser.add_argument('--shot', type=int, default=5,
                    help="Number of shot, for phase 2 only.")
parser.add_argument('--init-iter', type=int, default=50,
                    help="Number of iterations for OBJ(Target) initialization")
parser.add_argument('-max', '--max-iter', type=int, default=180000,
                    help='Number of training iterations.')
parser.add_argument('-b', '--batch-size', type=int, default=64,
                    help='Batch size for training')
parser.add_argument('--lr', '--learning-rate', type=float, default=4e-3,
                    help='Initial learning rate')
parser.add_argument('--steps', type=int, nargs='+', default=[120000, 150000],
                    help='Learning rate decrease steps.')
parser.add_argument('--warmup-iter', type=int, default=5000,
                    help='Batch size for training')
parser.add_argument('--ngpu', type=int, default=4, help='gpus')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Use cuda to train model')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='Gamma update for SGD')
parser.add_argument('--load-file', default=None,
                    help='Model checkpoint for loading.')
parser.add_argument('--resume', action='store_true',
                    help='Whether resume from the last checkpoint.'
                         'If True, no need to specify --load-file.')

# TODO
# Mixup
parser.add_argument('--mixup', action='store_true',
                    help='Whether to enable mixup.')
parser.add_argument('--no-mixup-iter', type=int, default=800,
                    help='Disable mixup for the last few iterations.')

# Output
parser.add_argument('--log', action='store_true',
                    help='Whether to log training details.')
parser.add_argument('--save-folder', default='./weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--checkpoint-period', type=int, default=10000,
                    help='Checkpoint period.')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

logger = setup_logger(args.save_folder)

if args.dataset == 'VOC':
    if args.setting == 'incre' and args.phase == 2:
        train_sets = [('2007', 'trainval')]
    else:
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
elif args.dataset == 'COCO':
    train_sets = [('2014', 'split_nonvoc_train'), ('2014', 'split_nonvoc_valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

if args.phase == 1:
    if args.dataset == 'VOC':
        src_cls_dim = 15
        num_classes = 16  # include background
    elif args.dataset == 'COCO':
        src_cls_dim = 60
        num_classes = 61  # include background
elif args.phase == 2:
    if args.setting == 'transfer':
        if args.method == 'ours':
            src_cls_dim = 60
            num_classes = 21
        elif args.method == 'ft':
            src_cls_dim = 20
            num_classes = 21
        else:
            raise ValueError(f"Unknown method: {args.method}")
    elif args.setting == 'incre':
        if args.method == 'ours':
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
p = 0.6
overlap_threshold = 0.5
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()
num_priors = priors.size(0)


def train(model, resume=False):
    model.train()
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    checkpointer = DetectionCheckpointer(
        model, args, optimizer=optimizer, scheduler=scheduler
    )
    criterion = MultiBoxLoss_combined(num_classes, overlap_threshold, True, 0, True, 3, 0.5, False)
    start_iter = (
        checkpointer.resume_or_load(args.basenet if args.phase == 1 else args.load_file,
                                    resume=resume).get("iteration", -1) + 1
    )
    max_iter = args.max_iter
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, args.checkpoint_period, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            TensorboardXWriter(args.save_folder),
        ]
    )

    if args.dataset == 'VOC':
        dataset = VOCDetection(args, VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform(0 if args.setting == 'transfer' else args.split))
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.phase == 2 and args.method == 'ours':
        sampler = TrainingSampler(len(dataset))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=detection_collate,
        )
        # initialize the OBJ(Target) parameters
        init_reweight(args, model, data_loader)
        dataset.set_mixup(np.random.beta, 1.5, 1.5)
        logger.info('Fine tuning on ' + str(args.shot) + '-shot task')

    sampler = TrainingSampler(len(dataset))
    data_loader = iter(torch.utils.data.DataLoader(
        dataset,
        args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
    ))
    assert model.training, 'Model.train() must be True during training.'
    logger.info("Starting training from iteration {}".format(start_iter))

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.gamma, last_epoch=epoch - 1)

    with EventStorage(start_iter) as storage:
        for iteration in range(start_iter, max_iter):
            iteration = iteration + 1
            storage.step()
            if args.phase == 2 and args.method == 'ours' and \
                    iteration == (args.max_iter - args.no_mixup_iter):
                dataset.set_mixup(None)
                data_loader = iter(torch.utils.data.DataLoader(
                    dataset,
                    args.batch_size,
                    sampler=sampler,
                    num_workers=args.num_workers,
                    collate_fn=detection_collate,
                ))

            data, targets = next(data_loader)
            # storage.put_image('image', vis_tensorboard(data))
            output = model(data)
            loss_dict = criterion(output, priors, targets)
            losses = sum(loss for loss in loss_dict.values())
            # assert torch.isfinite(losses).all(), loss_dict
            storage.put_scalars(total_loss=losses, **loss_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if args.phase == 2 and args.method == 'ours':
                if isinstance(model, (DistributedDataParallel, DataParallel)):
                    model.module.normalize()
                else:
                    model.normalize()
            storage.put_scalar("lr", optimizer.param_groups[-1]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def vis_tensorboard(images):
    rgb_mean = torch.Tensor(rgb_means).to(images.device)
    image = images[0] + rgb_mean[:, None, None]
    image = image[[2, 1, 0]].byte()
    return image


def init_reweight(args, model, data_loader):
    """
    Initialize the OBJ(Target) parameters.
    """
    logger.info('Initializing the OBJ(Target) parameters...')
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    cls_list = [torch.empty(0).to(device) for _ in range(num_classes-1)]

    for (data, targets), iteration in zip(data_loader, range(args.init_iter)):
        # vis_picture(images, targets)
        num = data.size(0)
        targets = [anno.to(device) for anno in targets]
        with torch.no_grad():
            conf_data = model(data, init=True)

        loc_t = torch.Tensor(num, num_priors, 4).to(device)
        conf_t = torch.Tensor(num, num_priors, 2).to(device)
        obj_t = torch.BoolTensor(num, num_priors).to(device)

        # match priors with gt
        for idx in range(num):  # batch_size
            truths = targets[idx][:, :-2].data  # [obj_num, 4]
            labels = targets[idx][:, -2:].data  # [obj_num]
            defaults = priors.data  # [num_priors,4]
            match(overlap_threshold, truths, defaults, [0.1, 0.2], labels, loc_t, conf_t, obj_t, idx)

        conf_data_list = [conf_data[conf_t[:, :, 0] == i] for i in range(1, num_classes)]
        cls_list = [torch.cat((cls_list[i], conf_data_list[i]), 0) for i in range(num_classes-1)]
    cls_list = [(item / item.norm(dim=1, keepdim=True)).mean(0) for item in cls_list]
    if args.setting == 'incre':
        cls_list = cls_list[15:]
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model.module.OBJ_Target.weight.data = torch.stack([item / item.norm() for item in cls_list], 0)
    else:
        model.OBJ_Target.weight.data = torch.stack([item / item.norm() for item in cls_list], 0)


if __name__ == '__main__':
    model = build_net(args, img_dim, src_cls_dim)
    logger.info("Model:\n{}".format(model))
    if args.cuda and torch.cuda.is_available():
        model.device = 'cuda'
        model.cuda()
        cudnn.benchmark = True
        if args.ngpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    else:
        model.device = 'cpu'
    train(model, args.resume)
