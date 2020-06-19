import torch
from torch.autograd import Function
from utils.box_utils import decode


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.variance = cfg['variance']

    def forward(self, predictions, prior):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [batch,num_priors,4]
        """

        loc, conf, obj = predictions

        loc_data = loc.data           # [num, num_priors, 4]
        conf_data = conf.data         # [num, num_priors, num_classes]
        prior_data = prior.data       # [num_priors, 4]
        obj_data = obj.data           # [num, num_priors, 2]
        num = loc_data.size(0)  # batch size
        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(num, self.num_priors, 4)
        self.scores = torch.zeros(num, self.num_priors, self.num_classes) # num_classes = 21
        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.scores = self.scores.cuda()

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_data[i].clone()

            conf_scores = obj_data[i, :, 1].unsqueeze(1).expand_as(conf_scores).mul(conf_scores) # conf score multiply the obj score
                                                                                                 # [num_priors, num_classes-1]
            conf_scores = torch.cat((obj_data[i, :, 0].unsqueeze(1), conf_scores), 1) # concatenate the background score
                                                                                      # [num_priors, num_classes]
            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        return self.boxes, self.scores
