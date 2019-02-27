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


    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, net):
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
        self.nonlinear = net.nonlinear
        self.scale = net.scale

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
        # torch.cuda.set_device(7)
        s_loc_data, s_conf_data, s_obj_data, q_loc_data, q_conf_data, q_obj_data = predictions
        s_t, q_t = targets
        n_way = s_conf_data.size(0)
        n_shot = s_conf_data.size(1)
        n_query = q_conf_data.size(1)
        num_priors = priors.size(0)
        num = n_way * (n_shot + n_query)
        s_loc_data = s_loc_data.view(-1, num_priors, 4)
        q_loc_data = q_loc_data.view(-1, num_priors, 4)
        # s_conf_data = s_conf_data.view(-1, num_priors, self.num_classes)
        # q_conf_data = q_conf_data.view(-1, num_priors, self.num_classes)
        s_obj_data = s_obj_data.view(-1, num_priors, 2)
        q_obj_data = q_obj_data.view(-1, num_priors, 2)

        # match priors (default boxes) and ground truth boxes
        if GPU:
            s_loc_t = torch.Tensor(n_way*n_shot, num_priors, 4).cuda()
            s_conf_t = torch.CharTensor(n_way*n_shot, num_priors).cuda()
            s_obj_t = torch.CharTensor(n_way*n_shot, num_priors).cuda()
            q_loc_t = torch.Tensor(n_way*n_query, num_priors, 4).cuda()
            q_conf_t = torch.CharTensor(n_way*n_query, num_priors).cuda()
            q_obj_t = torch.CharTensor(n_way*n_query, num_priors).cuda()

        else:
            s_loc_t = torch.Tensor(n_way * n_shot, num_priors, 4)
            s_conf_t = torch.CharTensor(n_way * n_shot, num_priors)
            s_obj_t = torch.ByteTensor(n_way * n_shot, num_priors)
            q_loc_t = torch.Tensor(n_way * n_query, num_priors, 4)
            q_conf_t = torch.CharTensor(n_way * n_query, num_priors)
            q_obj_t = torch.ByteTensor(n_way * n_query, num_priors)

        # match priors with gt for the support set
        for idx in range(n_way): # batch_size
            for idy in range(n_shot):
                truths = s_t[idx][idy][:, :-1].data  # [obj_num, 4]
                labels = s_t[idx][idy][:, -1].data   # [obj_num]
                defaults = priors.data             # [num_priors,4]
                match(self.threshold,truths,defaults,self.variance,labels,s_loc_t,s_conf_t,s_obj_t,idx*n_shot+idy)

        # match priors with gt for the query set
        for idx in range(n_way): # batch_size
            for idy in range(n_query):
                truths = q_t[idx][idy][:, :-1].data  # [obj_num, 4]
                labels = (idx+1)*torch.ones(truths.size(0))   # [obj_num]
                labels[q_t[idx][idy][:, -1] == -1] = -1
                defaults = priors.data             # [num_priors,4]
                match(self.threshold,truths,defaults,self.variance,labels,q_loc_t,q_conf_t,q_obj_t,idx*n_query+idy)

        s_pos = (s_conf_t>0).byte() # [n_way*n_shot, num_priors]
        q_pos = (q_conf_t>0).byte() # [n_way*n_query, num_priors]
        obj_t = torch.cat((s_obj_t, q_obj_t), 0).byte() # [num, num_priors]

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        loc_data = torch.cat((s_loc_data, q_loc_data), 0) # [num, num_priors, 4]
        loc_t = torch.cat((s_loc_t, q_loc_t), 0) # [num, num_priors, 4]
        obj_idx = obj_t.unsqueeze(obj_t.dim()).expand_as(loc_data)
        loc_p = loc_data[obj_idx].view(-1, 4) #整个batch的正样本priors
        loc_t = loc_t[obj_idx].view(-1, 4)    #对应的target
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') # size_average=False等价于reduction='sum'



        # Confidence Loss(cosine distance to classes center)
        # s_pos [n_way*n_shot, num_priors]
        # s_conf_data [n_way, n_shot, num_priors, num_classes]
        # with torch.no_grad():
        s_pos_index = s_pos.view(n_way, n_shot, num_priors).unsqueeze(3).expand_as(s_conf_data)  # [n_way, n_shot, num_priors, num_class]
        s_conf_data_list = [s_conf_data[i][s_pos_index[i]].view(-1, self.num_classes) for i in range(n_way)]

        # # method 1
        # way_list = []
        # for item in s_conf_data_list:
        #     item_list = []
        #     for j in range(item.size(0)):
        #         maped = self.nonlinear(item[j])
        #         item_list.append(maped/torch.norm(maped))
        #     tmp = torch.stack(item_list, 0).mean(0)
        #     way_list.append(tmp/torch.norm(tmp))
        # s_conf_cls_mean1 = torch.stack(way_list, 0) # [n_way, num_classes]

        # method 2 (faster than method 1)
        tmp_list = [torch.stack([self.nonlinear(item[j]) for j in range(item.size(0))], 0) for item in s_conf_data_list]
        tmp_list = [(item/torch.norm(item, dim=1, keepdim=True)).mean(0) for item in tmp_list]
        # tmp_list = [(item/torch.norm(item, dim=1, keepdim=True)).mean(0) for item in s_conf_data_list]
        s_conf_cls_mean = torch.stack([item/torch.norm(item) for item in tmp_list], 0) # [n_way, num_classes]

        q_conf_data = q_conf_data/torch.norm(q_conf_data, dim=3, keepdim=True) # [n_way, n_query, num_priors, num_classes]
        batch_conf = q_conf_data.view(-1, self.num_classes).mm(s_conf_cls_mean.t()) * self.scale # [n_way, num_classes]

        # Compute max conf across batch for hard negative mining (logit-combined)
        batch_obj = q_obj_data.view(-1, 2)  # [n_way*n_query*num_priors, 2]
        logit_0 = batch_obj[:, 0].unsqueeze(1) + torch.log(
            torch.exp(batch_conf).sum(dim=1, keepdim=True)) # [n_way*n_query*num_priors, 1]
        logit_k = batch_obj[:, 1].unsqueeze(1).expand_as(batch_conf) + batch_conf # [n_way*n_query*num_priors, n_way]
        logit = torch.cat((logit_0, logit_k), 1) # [n_way*n_query*num_priors, n_way+1]
        q_conf_tmp = q_conf_t.clone()
        q_conf_tmp[q_conf_tmp == -1] = 0 # set other boxes to bg and then filter out next
        with torch.no_grad():
            loss_c = F.cross_entropy(logit, q_conf_tmp.long().view(-1), reduction='none') # [n_way*n_query*num_priors]

        with torch.no_grad():
            # Hard Negative Mining
            loss_c[q_obj_t.byte().view(-1)] = 0 # filter out obj boxes for now
            loss_c = loss_c.view(n_way*n_query, -1) # [n_way*n_query, num_priors]
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = q_pos.long().sum(1, keepdim=True) # [n_way*n_query, 1]
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=num_priors - 1) # [n_way*n_query, 1]
            q_neg = idx_rank < num_neg.expand_as(idx_rank) # [n_way*n_query, num_priors]

        # Confidence Loss Including Positive and Negative Examples
        logit = logit.view(n_way*n_query, -1, n_way+1)
        pos_idx = q_pos.unsqueeze(2).expand_as(logit)
        neg_idx = q_neg.unsqueeze(2).expand_as(logit)
        mask = q_conf_t.ge(0).unsqueeze(2).expand_as(logit)
        # num1 = torch.sum((pos_idx + neg_idx).gt(0))/5           # for debug
        # num2 = torch.sum((pos_idx + neg_idx).gt(0).mul(mask))/5 # for debug
        conf_p = logit[(pos_idx + neg_idx).gt(0).mul(mask)].view(-1, n_way+1)
        conf_t = q_conf_t[(q_pos + q_neg).gt(0).mul(q_conf_t.ge(0))].long()
        loss_c = F.cross_entropy(conf_p, conf_t, reduction='sum')



        # Compute object loss across batch for hard negative mining
        obj_data = torch.cat((s_obj_data, q_obj_data), 0) # [num, num_priors, 2]
        with torch.no_grad():
            loss_obj = F.cross_entropy(obj_data.view(-1, 2), obj_t.long().view(-1), reduction='none') # [batch*num_priors]

        with torch.no_grad():
            # Hard Negative Mining
            loss_obj[obj_t.view(-1)] = 0 # filter out pos boxes for now, [num*num_priors]
            loss_obj = loss_obj.view(num, -1) # [num, num_priors]
            _, loss_idx = loss_obj.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = obj_t.long().sum(1, keepdim=True) # [num, 1] 每张图有多少个正类priors
            num_neg = torch.clamp(self.negpos_ratio*num_pos, max=num_priors-1) # [num, 1]
            neg = idx_rank < num_neg.expand_as(idx_rank) # [num, num_priors] 每张图里取loss_obj最大的num_neg个框用来计算loss_obj

        # Object Loss Including Positive and Negative Examples
        pos_idx = obj_t.unsqueeze(2).expand_as(obj_data) # [num, num_priors, 2]
        neg_idx = neg.unsqueeze(2).expand_as(obj_data) # [num, num_priors, 2]
        obj_p = obj_data[(pos_idx+neg_idx).gt(0)].view(-1, 2)
        obj_t = obj_t[(obj_t+neg).gt(0)]
        loss_obj = F.cross_entropy(obj_p, obj_t.long(), reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        sq_N = num_pos.data.sum().float()
        q_N = q_pos.data.sum().float()
        loss_l /= sq_N
        loss_c /= q_N
        loss_obj /= sq_N

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