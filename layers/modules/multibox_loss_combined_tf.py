import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
GPU = False
if torch.cuda.is_available():
    GPU = True


class MultiBoxLoss_combined(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss_combined, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, obj_data = predictions # conf_data[batch_size, num_priors, num_classes]
                                                    # loc_data[batch_size, num_priors, 4]
                                                    # obj_data[batch_size, num_priors, 2]
        priors = priors                             # shape[num_priors, 4]
        num = loc_data.size(0)                      # batch_size
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        if GPU:
            loc_t = torch.Tensor(num, num_priors, 4).cuda()
            conf_t = torch.CharTensor(num, num_priors).cuda()
            obj_t = torch.ByteTensor(num, num_priors).cuda()
        else:
            loc_t = torch.Tensor(num, num_priors, 4)
            conf_t = torch.CharTensor(num, num_priors)
            obj_t = torch.ByteTensor(num, num_priors)

        # match priors with gt
        for idx in range(num): # batch_size
            truths = targets[idx][:, :-1].data  # [obj_num, 4]
            labels = targets[idx][:, -1].data   # [obj_num]
            defaults = priors.data              # [num_priors,4]
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, obj_t, idx)

        pos = obj_t.byte() # [num, num_priors]

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4) #整个batch的正样本priors
        loc_t = loc_t[pos_idx].view(-1, 4)    #对应的target
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') # size_average=False等价于reduction='sum'

        # Compute max conf across batch for hard negative mining (logit-combined)
        batch_conf = conf_data.view(-1, self.num_classes)
        batch_obj = obj_data.view(-1, 2)
        logit_0 = batch_obj[:, 0].unsqueeze(1) + torch.log(
            torch.exp(batch_conf).sum(dim=1, keepdim=True))  # [num*num_priors, 1]
        logit_k = batch_obj[:, 1].unsqueeze(1).expand_as(batch_conf) + batch_conf
        logit = torch.cat((logit_0, logit_k), 1)
        loss_c = F.cross_entropy(logit, conf_t.long().view(-1), reduction='none') # [num*num_priors]

        # Hard Negative Mining
        loss_c[pos.view(-1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=num_priors - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        logit = logit.view(num, -1, self.num_classes+1)
        pos_idx = pos.unsqueeze(2).expand_as(logit)
        neg_idx = neg.unsqueeze(2).expand_as(logit)
        conf_p = logit[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes+1)
        conf_t = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, conf_t.long(), reduction='sum')

        # Compute object loss across batch for hard negative mining
        obj_p = obj_data.view(-1,2)
        loss_obj = F.cross_entropy(obj_p, obj_t.long().view(-1), reduction='none') # [batch*num_priors]

        # Hard Negative Mining
        loss_obj[pos.view(-1)] = 0 # filter out pos boxes for now
        loss_obj = loss_obj.view(num, -1)
        _, loss_idx = loss_obj.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True) # [batch, 1] 每个图有多少个正类priors
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=num_priors-1) #
        neg = idx_rank < num_neg.expand_as(idx_rank) # [batch, num_priors] 每张图里取loss_obj最大的num_neg个框用来计算loss_obj

        # Object Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(obj_data)
        neg_idx = neg.unsqueeze(2).expand_as(obj_data)
        obj_p = obj_data[(pos_idx+neg_idx).gt(0)].view(-1, 2)
        obj_t = obj_t[(pos+neg).gt(0)]
        loss_obj = F.cross_entropy(obj_p, obj_t.long(), reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N
        loss_obj /= N

        return loss_l, loss_c, loss_obj
        # return loss_l, loss_c

def vis_picture(imgs, targets):
    from data.coco_voc_form import COCO_CLASSES
    from utils.box_utils import point_form
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

    for i in range(n_way):
        cls = COCO_CLASSES[int(targets[i][0][targets[i][0][:, -1]!=-1, -1][0])]
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