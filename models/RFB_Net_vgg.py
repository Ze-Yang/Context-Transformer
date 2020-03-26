import torch
import torch.nn as nn
import os
import torch.nn.init as init


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

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
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

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


# RFB-s
class BasicRFB_a(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
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
                BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


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

    def __init__(self, args, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.method = args.method
        self.phase = args.phase
        self.setting = args.setting
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
        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.obj = nn.ModuleList(head[2])
        self.init_weight()
        if args.method == 'ours' and args.phase == 2:
            if args.setting == 'transfer':
                self.theta = nn.Linear(60, 60)
                self.phi = nn.Linear(60, 60)
                self.g = nn.Linear(60, 60)
                self.Wz = nn.Parameter(torch.FloatTensor(60))
                self.OBJ_Target = nn.Linear(60, 20, bias=False)
                self.scale = nn.Parameter(torch.FloatTensor([5]), requires_grad=False)
                init.kaiming_normal_(self.theta.weight, mode='fan_out')
                init.kaiming_normal_(self.phi.weight, mode='fan_out')
                init.kaiming_normal_(self.g.weight, mode='fan_out')
                self.theta.bias.data.fill_(0)
                self.phi.bias.data.fill_(0)
                self.g.bias.data.fill_(0)
                self.Wz.data.fill_(0)
            elif args.setting == 'incre':
                self.fc_base = nn.Linear(15, 15)
                self.theta = nn.Linear(15, 15)
                self.phi = nn.Linear(15, 15)
                self.g = nn.Linear(15, 15)
                self.Wz = nn.Parameter(torch.FloatTensor(15))
                self.OBJ_Target = nn.Linear(15, 5, bias=False)
                self.scale = nn.Parameter(torch.FloatTensor([5]), requires_grad=False)
                self.fc_base.weight.data.fill_(0)
                init.kaiming_normal_(self.theta.weight, mode='fan_out')
                init.kaiming_normal_(self.phi.weight, mode='fan_out')
                init.kaiming_normal_(self.g.weight, mode='fan_out')
                self.fc_base.bias.data.fill_(0)
                self.theta.bias.data.fill_(0)
                self.phi.bias.data.fill_(0)
                self.g.bias.data.fill_(0)
                self.Wz.data.fill_(0)

    def forward(self, x, init=False):
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
        x = x.to(self.device)
        num = x.size(0)
        sources = list()
        loc = list()
        conf = list()
        obj = list()
        conf_pool = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)  # a module that includes Relu
            if k < self.indicator or k%2 ==0:
                sources.append(x)

        kernel_size = [3, 2, 2, 2, 1, 1]
        stride = [3, 2, 2, 2, 1, 1]
        # apply multibox head to source layers
        for i, (x, l, c, o) in enumerate(zip(sources, self.loc, self.conf, self.obj)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())   # [num, map_size, map_size, 6*4]
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())  # [num, map_size, map_size, 6*num_classes]
            obj.append(o(x).permute(0, 2, 3, 1).contiguous())   # [num, map_size, map_size, 6*2]
            if self.method == 'ours' and self.phase == 2:
                conf_pool.append(nn.functional.max_pool2d(c(x), kernel_size=kernel_size[i], stride=stride[i],
                                                          ceil_mode=True).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # concat all the output feature maps
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        obj = torch.cat([o.view(o.size(0), -1) for o in obj], 1)

        if init:
            return conf.view(num, -1, self.num_classes)

        if self.method == 'ours' and self.phase == 2:
            conf_pool = torch.cat([o.view(o.size(0), -1) for o in conf_pool], 1)
            conf = conf.view(num, -1, self.num_classes)
            conf_pool = conf_pool.view(num, -1, self.num_classes)
            if self.setting == 'incre':
                conf_base = self.fc_base(conf) + conf
            conf_theta = self.theta(conf) + conf
            conf_phi = self.phi(conf_pool) + conf_pool
            conf_g = self.g(conf_pool) + conf_pool
            weight = torch.matmul(conf_theta, conf_phi.transpose(1, 2))
            weight = nn.functional.softmax(weight, dim=2)
            delta_conf = torch.matmul(weight, conf_g) * self.Wz
            conf_novel = conf + delta_conf
            conf_novel = conf_novel / conf_novel.norm(dim=2, keepdim=True)  # [num, num_priors, feature_dim]
            conf_novel = self.OBJ_Target(conf_novel) * self.scale  # [num*num_priors, 20]
            if self.setting == 'transfer':
                conf = conf_novel
            elif self.setting == 'incre':
                conf = torch.cat((conf_base, conf_novel), dim=2)

        if self.training:
            output = (
                loc.view(num, -1, 4),
                conf if self.phase == 2 and self.method == 'ours' else conf.view(num, -1, self.num_classes),
                obj.view(num, -1, 2)
            )
        else:
            output = (
                loc.view(num, -1, 4),
                nn.functional.softmax(
                    conf if self.phase == 2 and self.method == 'ours' else conf.view(num, -1, self.num_classes), dim=-1),
                nn.functional.softmax(obj.view(num, -1, 2), dim=-1)
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def init_weight(self):
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        # initialize newly added layers' weights with kaiming_normal method
        self.base.apply(weights_init)
        self.Norm.apply(weights_init)
        self.extras.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)
        self.obj.apply(weights_init)

    def normalize(self):
        self.OBJ_Target.weight.data = \
            self.OBJ_Target.weight / self.OBJ_Target.weight.norm(dim=1, keepdim=True)


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


def add_extras(size, cfg, in_channels):  # in_channels = 1024
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
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=4, stride=1, padding=1)]
    elif size == 300:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers


extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256, 'S', 256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    obj_layers = []
    vgg_source = [-2]  # the last but one layer of vgg network
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(512, cfg[k] * num_classes, kernel_size=3, padding=1)]
            obj_layers += [nn.Conv2d(512, cfg[k] * 2, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
            obj_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 2, kernel_size=3, padding=1)]
    i = 1
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k % 2 == 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]*4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]*num_classes, kernel_size=3, padding=1)]
            obj_layers += [nn.Conv2d(v.out_channels, cfg[i]*2, kernel_size=3, padding=1)]
            i += 1
    return vgg, extra_layers, (loc_layers, conf_layers, obj_layers)


mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(args, size, num_classes):
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return RFBNet(args, size, *multibox(size, vgg(base[str(size)], 3),
                                        add_extras(size, extras[str(size)], 1024),
                                        mbox[str(size)], num_classes), num_classes)
