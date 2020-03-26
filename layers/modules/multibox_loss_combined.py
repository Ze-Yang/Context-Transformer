import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match


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
        # loc_data[batch_size, num_priors, 4]
        # conf_data[batch_size, num_priors, num_classes]
        # obj_data[batch_size, num_priors, 2]
        loc_data, conf_data, obj_data = predictions

        device = loc_data.device
        targets = [anno.to(device) for anno in targets]
        num = loc_data.size(0)
        num_priors = priors.size(0)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4).to(device)
        conf_t = torch.Tensor(num, num_priors, 2).to(device)
        obj_t = torch.BoolTensor(num, num_priors).to(device)

        # match priors with gt
        for idx in range(num): # batch_size
            truths = targets[idx][:, :-2].data  # [obj_num, 4]
            labels = targets[idx][:, -2:].data  # [obj_num]
            defaults = priors.data              # [num_priors,4]
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, obj_t, idx)

        pos = (conf_t[:, :, 0] > 0).bool()  # [num, num_priors]
        num_pos = (conf_t[:, :, 1] * pos.float()).sum(1, keepdim=True).long()

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        loc_p = loc_data[pos]
        loc_t = loc_t[pos]
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='none')
        weight_pos = conf_t[pos][:, 1]
        loss_l = torch.sum(torch.sum(loss_l, dim=1) * weight_pos)

        # Compute object loss across batch for hard negative mining
        with torch.no_grad():
            loss_obj = F.cross_entropy(obj_data.view(-1, 2), obj_t.long().view(-1), reduction='none')
            # Hard Negative Mining
            loss_obj[obj_t.view(-1)] = 0  # filter out pos boxes (label>0) and ignored boxes (label=-1) for now
            loss_obj = loss_obj.view(num, -1)
            _, loss_idx = loss_obj.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=num_priors - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)  # [num, num_priors]

        # Object Loss Including Positive and Negative Examples
        mask = pos | neg
        weight = conf_t[mask][:, 1]
        loss_obj = torch.sum(F.cross_entropy(obj_data[mask], obj_t[mask].long(), reduction='none') * weight)

        # Confidence Loss (cosine distance to classes center)
        # pos [num, num_priors]
        # conf_data [num, num_priors, feature_dim]
        batch_conf = conf_data.view(-1, self.num_classes-1)

        # Compute max conf across batch for hard negative mining (logit-combined)
        batch_obj = obj_data.view(-1, 2)  # [num*num_priors, 2]
        logit_0 = batch_obj[:, 0].unsqueeze(1) + torch.log(
            torch.exp(batch_conf).sum(dim=1, keepdim=True))
        logit_k = batch_obj[:, 1].unsqueeze(1).expand_as(batch_conf) + batch_conf
        logit = torch.cat((logit_0, logit_k), 1)

        # Confidence Loss Including Positive and Negative Examples
        logit = logit.view(num, -1, self.num_classes)
        loss_c = torch.sum(F.cross_entropy(logit[mask], conf_t[mask][:, 0].long(), reduction='none') * weight)

        N = num_pos.sum()
        loss_l /= N
        loss_c /= N
        loss_obj /= N

        return {'loss_box_reg': loss_l, 'loss_cls': loss_c, 'loss_obj': loss_obj}
