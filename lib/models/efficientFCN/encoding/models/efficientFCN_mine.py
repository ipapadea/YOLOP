###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import math
from .base import BaseNet
from .fcn import FCNHead
#from ..nn import SyncBatchNorm, Encoding, Mean, GlobalAvgPool2d
from ..nn import Mean, GlobalAvgPool2d

__all__ = ['efficientFCN', 'HGDModule', 'get_efficientfcn', 'get_efficientfcn_resnet50_pcontext',
           'get_efficientfcn_resnet101_pcontext', 'get_efficientfcn_resnet50_citys',
           'get_efficientfcn_resnet101_citys', 'get_efficientfcn_resnet50_ade',
           'get_efficientfcn_resnet101_ade']

class efficientFCN(BaseNet):
    def __init__(self, nclass, backbone, num_center=256, aux=True, norm_layer=None, **kwargs):
        super(efficientFCN, self).__init__(nclass, backbone, aux, dilated=False,
                                     norm_layer=torch.nn.BatchNorm2d, **kwargs)
        self.head = HGDecoder(2048, 2, num_center=num_center,
                            norm_layer=norm_layer,
                            up_kwargs=self._up_kwargs)
        #if aux:
        #    self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = F.interpolate(x[0], imsize, **self._up_kwargs)
        #x[2] = F.interpolate(x[2], imsize, **self._up_kwargs)
        #if self.aux:
        #    #auxout = self.auxlayer(features[2])
        #    #auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
        #    x[1] = F.interpolate(x[1], imsize, **self._up_kwargs)
        #    #x.append(x[1])
        #    #x[2] = F.interpolate(x[2], imsize, **self._up_kwargs)
        #    #x.append(x[2])
        return tuple(x)


class HGDModule(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(HGDModule, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat= nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center= nn.Sequential(
            nn.Conv2d(in_channels*3, center_channels, 1, bias=False),
            #norm_layer(out_channels),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels))
        self.norm_center= nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0 = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels , 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1 = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, norm_layer):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size() # 512,12,20
        n1, c1, h1, w1 = guide1.size() # 512,24,40
        n2, c2, h2, w2 = guide2.size() # 512, 48, 80
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)  # 512, 48, 80
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)  # 512, 48,80
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1) #m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 1024
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels = 256
        f_cat = f_cat.view(n, self.out_channels, h*w)  # n x out_channels x 240
        #f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h*w)  # n x center_channels x 240
        f_center_norm = self.norm_center(f_center_norm)  # Softmax
        #n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))  # n x out_channels x center_channels
        

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)  # n x out_channels
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)  # n x out_channels x h2 x w2

        ###################################
        #f_affinity = self.conv_affinity(guide_cat)
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (1536x48x80)
        guide_cat_conv = self.conv_affinity0(guide_cat)  # G - n x out_channels x h2 x w2
        guide_cat_value_avg = guide_cat_conv + value_avg  # n x out_channels x h2 x w2
        f_affinity = self.conv_affinity1(guide_cat_value_avg)  # W - n x center_channels x h2 x w2
        n_aff, c_ff, h_aff, w_aff = f_affinity.size()
        f_affinity = f_affinity.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        norm_aff = ((self.center_channels) ** -.5)
        #x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up = norm_aff * x_center.bmm(f_affinity)  # n x out_channels x h2*w2
        x_up = x_up.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde -  n x out_channels x h2 x w2
        x_up_cat = torch.cat([x_up, guide_cat_conv], 1)  # f_8_hat
        x_up_conv = self.conv_up(x_up_cat)
        outputs = (x_up_conv,)
        return outputs


class MultiHGDModule(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModule, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels * 3, center_channels, 1, bias=False),
            # norm_layer(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0_drivable = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_drivable = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_lane = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up_drivable = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_up_lane = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

        # for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, norm_layer):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()  # 512,12,20
        n1, c1, h1, w1 = guide1.size()  # 512,24,40
        n2, c2, h2, w2 = guide2.size()  # 512, 48, 80
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)  # 512, 48, 80
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)  # 512, 48,80
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)  # m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 1024
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels = 256
        f_cat = f_cat.view(n, self.out_channels, h * w)  # n x out_channels x 240
        # f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)  # n x center_channels x 240
        f_center_norm = self.norm_center(f_center_norm)  # Softmax
        # n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))  # n x out_channels x center_channels

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)  # n x out_channels
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)  # n x out_channels x h2 x w2

        ###################################
        # f_affinity = self.conv_affinity(guide_cat)
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (1536x48x80)
        guide_cat_conv_drivable = self.conv_affinity0_drivable(guide_cat)  # G_drivable - n x out_channels x h2 x w2
        guide_cat_conv_lane = self.conv_affinity0_lane(guide_cat)  # G_lane - n x out_channels x h2 x w2
        guide_cat_value_avg_drivable = guide_cat_conv_drivable + value_avg  # n x out_channels x h2 x w2
        guide_cat_value_avg_lane = guide_cat_conv_lane + value_avg  # n x out_channels x h2 x w2
        f_affinity_drivable = self.conv_affinity1_drivable(guide_cat_value_avg_drivable)  # W_drivable - n x center_channels x h2 x w2
        f_affinity_lane = self.conv_affinity1_lane(guide_cat_value_avg_lane)  # W_lane - n x center_channels x h2 x w2
        n_aff, c_ff, h_aff, w_aff = f_affinity_lane.size()
        f_affinity_drivable = f_affinity_drivable.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        f_affinity_lane = f_affinity_lane.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        norm_aff = ((self.center_channels) ** -.5)
        # x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up_drivable = norm_aff * x_center.bmm(f_affinity_drivable)  # n x out_channels x h2*
        x_up_lane = norm_aff * x_center.bmm(f_affinity_lane)  # n x out_channels x h2*w2
        x_up_drivable = x_up_drivable.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_drivable -  n x out_channels x h2 x w2
        x_up_lane = x_up_lane.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_lane -  n x out_channels x h2 x w2
        x_up_cat_drivable = torch.cat([x_up_drivable, guide_cat_conv_drivable], 1)  # f_8_hat_drivable
        x_up_cat_lane = torch.cat([x_up_lane, guide_cat_conv_lane], 1)  # f_8_hat_lane
        x_up_conv_drivable = self.conv_up_drivable(x_up_cat_drivable)
        x_up_conv_lane = self.conv_up_lane(x_up_cat_lane)
        outputs = (x_up_conv_drivable, x_up_conv_lane)
        return outputs


class MultiHGDModuleObjectDetection(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModuleObjectDetection, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels * 3, center_channels, 1, bias=False),
            # norm_layer(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0_drivable = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_drivable = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_lane = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up_drivable = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_up_lane = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()  # 512,12,20
        n1, c1, h1, w1 = guide1.size()  # 512,24,40
        n2, c2, h2, w2 = guide2.size()  # 512, 48, 80
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)  # 512, 48, 80
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)  # 512, 48,80
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)  # m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 1024
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels = 256
        f_cat = f_cat.view(n, self.out_channels, h * w)  # n x out_channels x 240
        # f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)  # n x center_channels x 240
        f_center_norm = self.norm_center(f_center_norm)  # Softmax
        # n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))  # n x out_channels x center_channels

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)  # n x out_channels
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)  # n x out_channels x h2 x w2

        ###################################
        # f_affinity = self.conv_affinity(guide_cat)
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (1536x48x80)
        guide_cat_conv_drivable = self.conv_affinity0_drivable(guide_cat)  # G_drivable - n x out_channels x h2 x w2
        guide_cat_conv_lane = self.conv_affinity0_lane(guide_cat)  # G_lane - n x out_channels x h2 x w2
        guide_cat_value_avg_drivable = guide_cat_conv_drivable + value_avg  # n x out_channels x h2 x w2
        guide_cat_value_avg_lane = guide_cat_conv_lane + value_avg  # n x out_channels x h2 x w2
        f_affinity_drivable = self.conv_affinity1_drivable(guide_cat_value_avg_drivable)  # W_drivable - n x center_channels x h2 x w2
        f_affinity_lane = self.conv_affinity1_lane(guide_cat_value_avg_lane)  # W_lane - n x center_channels x h2 x w2
        n_aff, c_ff, h_aff, w_aff = f_affinity_lane.size()
        f_affinity_drivable = f_affinity_drivable.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        f_affinity_lane = f_affinity_lane.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        norm_aff = ((self.center_channels) ** -.5)
        # x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up_drivable = norm_aff * x_center.bmm(f_affinity_drivable)  # n x out_channels x h2*
        x_up_lane = norm_aff * x_center.bmm(f_affinity_lane)  # n x out_channels x h2*w2
        x_up_drivable = x_up_drivable.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_drivable -  n x out_channels x h2 x w2
        x_up_lane = x_up_lane.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_lane -  n x out_channels x h2 x w2
        x_up_cat_drivable = torch.cat([x_up_drivable, guide_cat_conv_drivable], 1)  # f_8_hat_drivable
        x_up_cat_lane = torch.cat([x_up_lane, guide_cat_conv_lane], 1)  # f_8_hat_lane
        x_up_conv_drivable = self.conv_up_drivable(x_up_cat_drivable)
        x_up_conv_lane = self.conv_up_lane(x_up_cat_lane)
        outputs = (x_up_conv_drivable, x_up_conv_lane)
        return outputs

class MultiHGDModule_1632(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModule_1632, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels * 3, center_channels, 1, bias=False),
            # norm_layer(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False), ######
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_drivable = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False), ######
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1 = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

        # for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, norm_layer):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()  # 512,12,20
        n1, c1, h1, w1 = guide1.size()  # 512,24,40
        n2, c2, h2, w2 = guide2.size()  # 512, 48, 80
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)  # 512, 48, 80
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)  # 512, 48,80
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)  # m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 1024
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels = 256
        f_cat = f_cat.view(n, self.out_channels, h * w)  # n x out_channels x 240
        # f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)  # n x center_channels x 240
        f_center_norm = self.norm_center(f_center_norm)  # Softmax
        # n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))  # n x out_channels x center_channels

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)  # n x out_channels
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)  # n x out_channels x h2 x w2

        ###################################
        # f_affinity = self.conv_affinity(guide_cat)
        guide_cat_lane = torch.cat([guide2, x_up1, x_up0], 1)  # m8_lane - n x 3*in_channels x h2 x w2 (1536x48x80)
        guide_cat_drivable = torch.cat([x_up1, x_up0], 1)  # m8_drivable - n x 3*in_channels x h2 x w2 (1024x48x80)
        guide_cat_conv_lane = self.conv_affinity0_lane(guide_cat_lane)  # G_lane - n x out_channels x h2 x w2
        guide_cat_conv_drivable = self.conv_affinity0_drivable(guide_cat_drivable)  # G_drivable - n x out_channels x h2 x w2
        guide_cat_value_avg_lane = guide_cat_conv_lane + value_avg  # n x out_channels x h2 x w2
        guide_cat_value_avg_drivable = guide_cat_conv_drivable + value_avg  # n x out_channels x h2 x w2
        f_affinity_lane = self.conv_affinity1(guide_cat_value_avg_lane)  # W_lane - n x center_channels x h2 x w2
        f_affinity_drivable = self.conv_affinity1(guide_cat_value_avg_drivable)  # W_drivable - n x center_channels x h2 x w2
        n_aff, c_ff, h_aff, w_aff = f_affinity_lane.size()
        f_affinity_lane = f_affinity_lane.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        f_affinity_drivable = f_affinity_drivable.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        norm_aff = ((self.center_channels) ** -.5)
        # x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up_lane = norm_aff * x_center.bmm(f_affinity_lane)  # n x out_channels x h2*w2
        x_up_drivable = norm_aff * x_center.bmm(f_affinity_drivable)  # n x out_channels x h2*w2
        x_up_lane = x_up_lane.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_lane -  n x out_channels x h2 x w2
        x_up_drivable = x_up_drivable.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_drivable -  n x out_channels x h2 x w2
        x_up_cat_lane = torch.cat([x_up_lane, guide_cat_conv_lane], 1)  # f_8_hat_lane
        x_up_cat_drivable = torch.cat([x_up_drivable, guide_cat_conv_drivable], 1)  # f_8_hat_drivable
        x_up_conv_lane = self.conv_up(x_up_cat_lane)
        x_up_conv_drivable = self.conv_up(x_up_cat_drivable)
        outputs = (x_up_conv_lane, x_up_conv_drivable)
        return outputs


class HGDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(HGDecoder, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, padding=0, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(512, 256, 1, padding=0, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))
        #self.conv30 = nn.Sequential(
        #    nn.Conv2d(512, 512, 1, padding=0, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        #self.conv52 = nn.Sequential(
        #    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        #self.conv53 = nn.Sequential(
        #    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        #self.conv51 = nn.Sequential(
        #    nn.Conv2d((512+out_channels), 512, 1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        #self.num_center = 128
        #self.num_center = 256
        #self.num_center = int(out_channels * 4)
        #self.num_center = out_channels
        #self.num_center = 600
        self.num_center = num_center
        self.hgdmodule0 = HGDModule(256, self.num_center, 512, norm_layer=norm_layer)
        self.conv_pred3_lanes = nn.Sequential(nn.Dropout2d(0.1, False),
            nn.Conv2d(512, out_channels, 1, padding=0))
        self.conv_pred3_drivable = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(512, out_channels, 1, padding=0))
        # self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)
        #for m in self.modules():
        #    #print(f"initialize {m} layer.")
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, norm_layer):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

               

    def forward(self, *inputs):
        # feat50 = self.conv50(inputs[-1])
        # feat40 = self.conv40(inputs[-2])
        #feat30 = self.conv30(inputs[-3])
        feat_res2, feat_res4, feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)
        outs_pred3_lanes = self.conv_pred3_lanes(outs0[0])
        outs_pred3_drivable = self.conv_pred3_drivable(outs0[1])
        outs = (outs_pred3_lanes, outs_pred3_drivable)

        return outs


class MultiHGDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoder, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, padding=0, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(512, 256, 1, padding=0, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))
        # self.conv30 = nn.Sequential(
        #    nn.Conv2d(512, 512, 1, padding=0, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        # self.conv52 = nn.Sequential(
        #    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        # self.conv53 = nn.Sequential(
        #    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        # self.conv51 = nn.Sequential(
        #    nn.Conv2d((512+out_channels), 512, 1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        # self.num_center = 128
        # self.num_center = 256
        # self.num_center = int(out_channels * 4)
        # self.num_center = out_channels
        # self.num_center = 600
        self.num_center = num_center
        self.hgdmodule0 = MultiHGDModule(256, self.num_center, 512, norm_layer=norm_layer)
        # self.conv_pred3_lanes = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                       nn.Conv2d(512, out_channels, 1, padding=0))
        # self.conv_pred3_drivable = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                          nn.Conv2d(512, out_channels, 1, padding=0))
        # Adding 1x1 conv to adjust the number of channels
        self.conv1x1_1_drivable = nn.Conv2d(512, 256, 1)  # adjust the input and output channels appropriately
        self.conv1x1_2_drivable = nn.Conv2d(256, 128, 1)  # adjust the input and output channels appropriately
        self.conv1x1_3_drivable = nn.Conv2d(128, 32, 1)  # adjust the input and output channels appropriately

        self.conv1x1_1_lanes = nn.Conv2d(512, 256, 1)  # adjust the input and output channels appropriately
        self.conv1x1_2_lanes = nn.Conv2d(256, 128, 1)  # adjust the input and output channels appropriately
        self.conv1x1_3_lanes = nn.Conv2d(128, 32, 1)  # adjust the input and output channels appropriately
        # self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, *inputs):

        feat_res2, feat_res4, feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  # 1, 512, 48, 80
        # outs_pred3_lanes = self.conv_pred3_lanes(outs0[0])
        # outs_pred3_drivable = self.conv_pred3_drivable(outs0[0])
        outs_drivable = self.conv1x1_1_drivable(outs0[0])
        outs_drivable = self.conv1x1_2_drivable(outs_drivable)
        outs_drivable = self.conv1x1_3_drivable(outs_drivable)
        outs_lane = self.conv1x1_1_lanes(outs0[1])
        outs_lane = self.conv1x1_2_lanes(outs_lane)
        outs_lane = self.conv1x1_3_lanes(outs_lane)
        outs = (outs_drivable, outs_lane)

        return outs

class MultiHGDecoderObjectDetection(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 num_classes=80, num_anchors=3,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoderObjectDetection, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv50 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, padding=0, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(512, 256, 1, padding=0, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))
        self.num_center = num_center
        self.hgdmodule0 = MultiHGDModuleObjectDetection(256, self.num_center, 512, norm_layer=norm_layer)

        # Adding 1x1 conv to adjust the number of channels
        self.conv1x1_1_drivable = nn.Conv2d(512, 256, 1)  # adjust the input and output channels appropriately
        self.conv1x1_2_drivable = nn.Conv2d(256, 128, 1)  # adjust the input and output channels appropriately
        self.conv1x1_3_drivable = nn.Conv2d(128, 32, 1)  # adjust the input and output channels appropriately

        self.conv1x1_1_lanes = nn.Conv2d(512, 256, 1)  # adjust the input and output channels appropriately
        self.conv1x1_2_lanes = nn.Conv2d(256, 128, 1)  # adjust the input and output channels appropriately
        self.conv1x1_3_lanes = nn.Conv2d(128, 32, 1)  # adjust the input and output channels appropriately
        # self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)

        # Detection head
        self.detection_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (num_classes + 5) * num_anchors, 1)  # Output layer for detection
        )

    def forward(self, *inputs):

        feat_res2, feat_res4, feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)
        feat_16 = self.conv40(feat_res16)
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  # 1, 512, 48, 80
        outs_drivable = self.conv1x1_1_drivable(outs0[0])
        outs_drivable = self.conv1x1_2_drivable(outs_drivable)
        outs_drivable = self.conv1x1_3_drivable(outs_drivable)
        outs_lane = self.conv1x1_1_lanes(outs0[1])
        outs_lane = self.conv1x1_2_lanes(outs_lane)
        outs_lane = self.conv1x1_3_lanes(outs_lane)

        # Detection processing
        detection = self.detection_conv(outs0[2])  # Assuming outs0[0] is suitable for detection

        outs = (outs_drivable, outs_lane, detection)

        return outs

class MultiHGDecoder_1632(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoder_1632, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, padding=0, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(512, 256, 1, padding=0, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))
        # self.conv30 = nn.Sequential(
        #    nn.Conv2d(512, 512, 1, padding=0, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        # self.conv52 = nn.Sequential(
        #    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        # self.conv53 = nn.Sequential(
        #    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        # self.conv51 = nn.Sequential(
        #    nn.Conv2d((512+out_channels), 512, 1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        # self.num_center = 128
        # self.num_center = 256
        # self.num_center = int(out_channels * 4)
        # self.num_center = out_channels
        # self.num_center = 600
        self.num_center = num_center
        self.hgdmodule0 = MultiHGDModule_1632(256, self.num_center, 512, norm_layer=norm_layer)
        # self.conv_pred3_lanes = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                       nn.Conv2d(512, out_channels, 1, padding=0))
        # self.conv_pred3_drivable = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                          nn.Conv2d(512, out_channels, 1, padding=0))
        # Adding 1x1 conv to adjust the number of channels
        self.conv1x1_1_lanes = nn.Conv2d(512, 256, 1)  # adjust the input and output channels appropriately
        self.conv1x1_2_lanes = nn.Conv2d(256, 128, 1)  # adjust the input and output channels appropriately
        self.conv1x1_3_lanes = nn.Conv2d(128, 32, 1)  # adjust the input and output channels appropriately
        # self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv1x1_1_drivable = nn.Conv2d(512, 256, 1)  # adjust the input and output channels appropriately
        self.conv1x1_2_drivable = nn.Conv2d(256, 128, 1)  # adjust the input and output channels appropriately
        self.conv1x1_3_drivable = nn.Conv2d(128, 32, 1)  # adjust the input and output channels appropriately

    def forward(self, *inputs):

        feat_res2, feat_res4, feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  # 1, 512, 48, 80

        outs_lane = self.conv1x1_1_lanes(outs0[0])
        outs_lane = self.conv1x1_2_lanes(outs_lane)
        outs_lane = self.conv1x1_3_lanes(outs_lane)
        outs_drivable = self.conv1x1_1_drivable(outs0[1])
        outs_drivable = self.conv1x1_2_drivable(outs_drivable)
        outs_drivable = self.conv1x1_3_drivable(outs_drivable)
        outs = (outs_lane, outs_drivable)

        return outs


def get_efficientfcn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='~/.encoding/models', **kwargs):
    r"""EncNet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    #kwargs['lateral'] = True if dataset.lower().startswith('p') else False
    # infer number of classes
    from ..datasets import datasets, acronyms
    model = efficientFCN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('efficientFCN_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_efficientfcn_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('pcontext', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_efficientfcn_resnet101_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('pcontext', 'resnet101', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_efficientfcn_resnet50_citys(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_citys(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('citys', 'resnet50', pretrained, root=root, aux=True,
                      base_size=1024, crop_size=768, **kwargs)

def get_efficientfcn_resnet101_citys(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet101_citys(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('citys', 'resnet101', pretrained, root=root, aux=True,
                      base_size=1024, crop_size=768, **kwargs)


def get_efficientfcn_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('ade20k', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_efficientfcn_resnet101_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('ade20k', 'resnet101', pretrained, root=root, aux=True,
                      base_size=640, crop_size=576, **kwargs)

def get_efficientfcn_resnet152_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('ade20k', 'resnet152', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)
