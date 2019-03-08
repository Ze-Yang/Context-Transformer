import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# RFB
class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


# RFB-s
class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)), # stride一旦不取1,feature map无法concat
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)), # stride一旦不取1,feature map无法concat
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False) # stride一旦不取1,feature map无法concat
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


class l2_norm(nn.Module):
    def __init__(self, dim):
        super(l2_norm, self).__init__()
        self.dim = dim

    def forward(self, input):
        output = input / torch.norm(input, dim=self.dim, keepdim=True)
        return output


def composite_fc(bn, norm, fc):
    def fc_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        output = fc(norm(bn(concated_features)))
        return output

    return fc_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate): # bn_size: bottleneck_size
        super(_DenseLayer, self).__init__()
        self.add_module('bn', nn.BatchNorm1d(num_input_features)),
        self.add_module('norm', l2_norm(1)),
        # self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('fc', nn.Linear(num_input_features, growth_rate, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, *prev_features):
        fc_function = composite_fc(self.bn, self.norm, self.fc)
        new_features = fc_function(*prev_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.obj = nn.ModuleList(head[2])
        self.scale = nn.Parameter(torch.FloatTensor([10]))
        self.add_module('denselayer1', _DenseLayer(60, 20, 0.5))
        self.add_module('denselayer2', _DenseLayer(80, 20, 0.5))
        self.add_module('denselayer3', _DenseLayer(100, 20, 0.5))

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        obj = list()
        if len(x) == 2:
            meta_learning = True
            n_way = x[0].size(0)
            n_support = x[0].size(1)
            n_query = x[1].size(1)
            x = torch.cat(x, 1).view(-1, 3, 300, 300)
        else:
            meta_learning = False
            if x.dim()==5:
                n_way = x.size(0)
                per_way = x.size(1)
                x = x.view(-1, 3, 300, 300)
            else:
                n_way = None

        # apply vgg up to conv4_3 relu
        for k in range(23):
            # for param in self.base[k].parameters():
            #     a = param
            x = self.base[k](x)

        s = self.Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)                             # 包括ReLu了,是一个小module
            if k < self.indicator or k%2 ==0:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c, o) in zip(sources, self.loc, self.conf, self.obj):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())   # [num, map_size, map_size, 6*4]
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())  # [num, map_size, map_size, 6*num_classes]
            obj.append(o(x).permute(0, 2, 3, 1).contiguous())   # [num, map_size, map_size, 6*2]

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) # 把所有的feature map的输出拼合起来
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        obj = torch.cat([o.view(o.size(0), -1) for o in obj], 1)

        # 把所有的feature map的输出拼合起来
        if meta_learning:
            loc = loc.view(n_way, n_support + n_query, -1, 4)
            conf = conf.view(n_way, n_support + n_query, -1, self.num_classes)
            obj = obj.view(n_way, n_support + n_query, -1, 2)
            s_loc = loc[:, :n_support]   # [n_way, n_support, num_priors, 4]
            s_conf = conf[:, :n_support] # [n_way, n_support, num_priors, num_classes]
            s_obj = obj[:, :n_support]   # [n_way, n_support, num_priors, 2]
            q_loc = loc[:, n_support:]   # [n_way, n_query, num_priors, 4]
            q_conf = conf[:, n_support:] # [n_way, n_query, num_priors, num_classes]
            q_obj = obj[:, n_support:]   # [n_way, n_query, num_priors, 2]

            output = (s_loc, s_conf, s_obj, q_loc, q_conf, q_obj)
        else:
            if n_way:
                loc = loc.view(n_way, per_way, -1, 4)
                conf = conf.view(n_way, per_way, -1, self.num_classes)
                obj = obj.view(n_way, per_way, -1, 2)
            else:
                loc = loc.view(loc.size(0), -1, 4)
                conf = conf.view(conf.size(0), -1, self.num_classes)
                obj = obj.view(obj.size(0), -1, 2)

            output = (loc, conf, obj)

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def normalize(self):
        self.denselayer1.fc.weight.data = self.denselayer1.fc.weight / torch.norm(self.denselayer1.fc.weight, dim=1, keepdim=True)
        self.denselayer2.fc.weight.data = self.denselayer2.fc.weight / torch.norm(self.denselayer2.fc.weight, dim=1, keepdim=True)
        self.denselayer3.fc.weight.data = self.denselayer3.fc.weight / torch.norm(self.denselayer3.fc.weight, dim=1, keepdim=True)

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, in_channels, batch_norm=False): # in_channels = 1024
    # Extra layers added to VGG for feature scaling
    layers = []
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale=1.0, visual=1)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale=1.0, visual=2)]
            else:
                layers += [BasicRFB(in_channels, v, scale=1.0, visual=2)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    elif size ==300:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    obj_layers = []
    vgg_source = [-2] # vgg网络的倒数第二层
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
            obj_layers += [nn.Conv2d(512,
                                 cfg[k] * 2, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
            obj_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 2, kernel_size=3, padding=1)]
    i = 1
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2 == 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            obj_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 2, kernel_size=3, padding=1)]
            i += 1
    return vgg, extra_layers, (loc_layers, conf_layers, obj_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21, overlap_threshold=0.5):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return RFBNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
