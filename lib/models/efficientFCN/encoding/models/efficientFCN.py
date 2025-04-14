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
import torch.nn.utils.prune as prune

def apply_pruning(module, amount=0.5, method='ln_structured', n=2, dim=0):
    if isinstance(module, nn.Conv2d):
        if method == 'ln_structured':
            prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
        elif method == 'l1_unstructured':
            prune.l1_unstructured(module, name='weight', amount=amount)
        # Remove reparameterization to make pruning permanent
        prune.remove(module, 'weight')
def apply_pruning(module, amount=0.5, method='ln_structured', n=2, dim=0):
    if isinstance(module, nn.Conv2d):
        if method == 'ln_structured':
            prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
        elif method == 'l1_unstructured':
            prune.l1_unstructured(module, name='weight', amount=amount)
        # Remove reparameterization to make pruning permanent
        prune.remove(module, 'weight')

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


class MultiHGDModuleTwinLiteNetv2(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModuleTwinLiteNetv2, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels*3, center_channels, 1, bias=False), # in_channels * 3 evala 211
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
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
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
        n, c, h, w = x.size()  #
        n1, c1, h1, w1 = guide1.size()  #
        n2, c2, h2, w2 = guide2.size()  #
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)  # m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 64
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels =
        f_cat = f_cat.view(n, self.out_channels, h * w)  # n x out_channels x 240
        # f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)  # n x center_channels x 240
        f_center_norm = self.norm_center(f_center_norm)  # Softmax A~ (1)
        # n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))  # n x out_channels x center_channels   Ci (2)

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)  # n x out_channels
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)  # n x out_channels x h2 x w2

        ###################################
        # f_affinity = self.conv_affinity(guide_cat)
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (211x180x320)
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


class MultiHGDecoderTwinLiteNet2ObjectDetectionSingleTask(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoderTwinLiteNet2ObjectDetectionSingleTask, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(256, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(131, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv30 = nn.Sequential(
           nn.Conv2d(19, 64, 1, padding=0, bias=False),
           norm_layer(64),
           nn.ReLU(inplace=True))

        self.num_center = num_center
        self.hgdmodule0 = MultiHGDModuleObjectDetectionTwinLiteNetv2SingleTask(in_channels=64, center_channels=192, out_channels=32, norm_layer=norm_layer)#, self.num_center,out_channels = 32, norm_layer=norm_layer)

    def forward(self, *inputs):

        feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        feat_8 = self.conv30(feat_res8)
        # outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  #
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_8)  #

        return outs0


class MultiHGDModuleObjectDetectionTwinLiteNetv2SingleTask(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModuleObjectDetectionTwinLiteNetv2SingleTask, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels*3, center_channels, 1, bias=False), # in_channels * 3 evala 211
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0_object = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_object = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up_object = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()  #
        n1, c1, h1, w1 = guide1.size()  #
        n2, c2, h2, w2 = guide2.size()  #
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)  # m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 64
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels =
        f_cat = f_cat.view(n, self.out_channels, h * w)  # n x out_channels x 240
        # f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)  # n x center_channels x 240
        f_center_norm = self.norm_center(f_center_norm)  # Softmax A~ (1)
        # n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))  # n x out_channels x center_channels   Ci (2)

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)  # n x out_channels
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)  # n x out_channels x h2 x w2

        ###################################
        # f_affinity = self.conv_affinity(guide_cat)
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (211x180x320)
        guide_cat_conv_object = self.conv_affinity0_object(guide_cat)
        # print("output of guide_cat_conv_object shape: ", guide_cat_conv_object.shape)

        guide_cat_value_avg_object = guide_cat_conv_object + value_avg  # n x out_channels x h2 x w2
        f_affinity_object = self.conv_affinity1_object(guide_cat_value_avg_object)  # W_lane - n x center_channels x h2 x w2
        # print("output of f_affinity_object shape: ", f_affinity_object.shape)

        n_aff, c_ff, h_aff, w_aff = f_affinity_object.size()
        f_affinity_object = f_affinity_object.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        # print("output of f_affinity_object shape: ", f_affinity_object.shape)

        norm_aff = ((self.center_channels) ** -.5)
        # x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up_object = norm_aff * x_center.bmm(f_affinity_object)  # n x out_channels x h2*w2
        # print("output of x_up_object shape: ", x_up_object.shape)

        x_up_object = x_up_object.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_object -  n x out_channels x h2 x w2
        # print("output of x_up_object shape: ", x_up_object.shape)

        x_up_cat_object = torch.cat([x_up_object, guide_cat_conv_object], 1)  # f_8_hat_object
        # print("output of x_up_cat_object shape: ", x_up_cat_object.shape)

        x_up_conv_object = self.conv_up_object(x_up_cat_object)
        # print("output of x_up_conv_object shape: ", x_up_conv_object.shape)

        # outputs = (x_up_conv_drivable, x_up_conv_lane, x_up_conv_object)
        return x_up_conv_object#outputs


class MultiHGDModuleTwinLiteNetv2HalfCodewords(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModuleTwinLiteNetv2HalfCodewords, self).__init__()
        center_channels=center_channels//2
        out_channels=out_channels//2
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels*3, center_channels, 1, bias=False), # in_channels * 3 evala 211
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
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
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
        n, c, h, w = x.size()  #
        n1, c1, h1, w1 = guide1.size()  #
        n2, c2, h2, w2 = guide2.size()  #
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)  # m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 64
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels =
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
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (211x180x320)
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

class MultiHGDModuleTwinLiteNetv2QuarterCodewords(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModuleTwinLiteNetv2QuarterCodewords, self).__init__()
        center_channels=center_channels//4
        out_channels=out_channels//4
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels*3, center_channels, 1, bias=False), # in_channels * 3 evala 211
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
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
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
        n, c, h, w = x.size()  #
        n1, c1, h1, w1 = guide1.size()  #
        n2, c2, h2, w2 = guide2.size()  #
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)  # m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 64
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels =
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
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (211x180x320)
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

class MultiHGDModuleTwinLiteNetv2DoubleCodewords(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModuleTwinLiteNetv2DoubleCodewords, self).__init__()
        center_channels=center_channels*2
        out_channels=out_channels*2
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels*3, center_channels, 1, bias=False), # in_channels * 3 evala 211
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
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),  # in_channels * 3 evala 211
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
        n, c, h, w = x.size()  #
        n1, c1, h1, w1 = guide1.size()  #
        n2, c2, h2, w2 = guide2.size()  #
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)  # m32
        f_cat = self.conv_cat(x_cat)  # Base Feature Maps - out_channels = 64
        f_center = self.conv_center(x_cat)  # Spatial Weighting Maps - center_channels =
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
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (211x180x320)
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


class MultiHGDModuleTwinLiteNetv2Scaled(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, scale=1, norm_layer=None):
        super(MultiHGDModuleTwinLiteNetv2Scaled, self).__init__()
        self.in_channels = int(in_channels * scale)
        self.center_channels = int(center_channels * scale)
        self.out_channels = int(out_channels * scale)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0_drivable = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_drivable = nn.Sequential(
            nn.Conv2d(self.out_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_lane = nn.Sequential(
            nn.Conv2d(self.out_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True))
        self.conv_up_drivable = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_up_lane = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

        # Apply pruning
        # self.apply(lambda module: apply_pruning(module, amount=prune_amount))

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()
        # n1, c1, h1, w1 = guide1.size()
        n2, c2, h2, w2 = guide2.size()
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)
        f_cat = self.conv_cat(x_cat)
        f_center = self.conv_center(x_cat)
        f_cat = f_cat.view(n, self.out_channels, h * w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)
        f_center_norm = self.norm_center(f_center_norm)
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))

        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)

        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)
        guide_cat_conv_drivable = self.conv_affinity0_drivable(guide_cat)
        guide_cat_conv_lane = self.conv_affinity0_lane(guide_cat)
        guide_cat_value_avg_drivable = guide_cat_conv_drivable + value_avg
        guide_cat_value_avg_lane = guide_cat_conv_lane + value_avg
        f_affinity_drivable = self.conv_affinity1_drivable(guide_cat_value_avg_drivable)
        f_affinity_lane = self.conv_affinity1_lane(guide_cat_value_avg_lane)
        n_aff, c_ff, h_aff, w_aff = f_affinity_lane.size()
        f_affinity_drivable = f_affinity_drivable.view(n_aff, c_ff, h_aff * w_aff)
        f_affinity_lane = f_affinity_lane.view(n_aff, c_ff, h_aff * w_aff)
        norm_aff = ((self.center_channels) ** -.5)
        x_up_drivable = norm_aff * x_center.bmm(f_affinity_drivable)
        x_up_lane = norm_aff * x_center.bmm(f_affinity_lane)
        x_up_drivable = x_up_drivable.view(n, self.out_channels, h_aff, w_aff)
        x_up_lane = x_up_lane.view(n, self.out_channels, h_aff, w_aff)

        # Free memory before concatenation
        # del guide_cat_value_avg_drivable, guide_cat_value_avg_lane, f_affinity_drivable, f_affinity_lane, x_center
        # torch.cuda.empty_cache()
        x_up_cat_drivable = torch.cat([x_up_drivable, guide_cat_conv_drivable], 1)
        x_up_cat_lane = torch.cat([x_up_lane, guide_cat_conv_lane], 1)
        x_up_conv_drivable = self.conv_up_drivable(x_up_cat_drivable)
        x_up_conv_lane = self.conv_up_lane(x_up_cat_lane)
        outputs = (x_up_conv_drivable, x_up_conv_lane)
        return outputs

class MultiHGDModuleTwinLiteNetv2ScaledObjectDetection(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, scale=1, norm_layer=None):
        super(MultiHGDModuleTwinLiteNetv2ScaledObjectDetection, self).__init__()
        self.in_channels = int(in_channels * scale)
        self.center_channels = int(center_channels * scale)
        self.out_channels = int(out_channels * scale)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0_drivable = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_lane = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity0_object = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_drivable = nn.Sequential(
            nn.Conv2d(self.out_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_lane = nn.Sequential(
            nn.Conv2d(self.out_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1_object = nn.Sequential(
            nn.Conv2d(self.out_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, 1, bias=False),
            norm_layer(self.center_channels),
            nn.ReLU(inplace=True))
        self.conv_up_drivable = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_up_lane = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.conv_up_object = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

        # Object Detection Feature Convolutions (OD Heads)
        self.conv_od2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),  # Channel transformation
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # Downsample to (96x160)
            norm_layer(64),
            nn.ReLU(inplace=True)
        )

        # ✅ od_N3 (96x160) → (48x80)
        self.conv_od3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0, bias=False),  # Channel transformation
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # Downsample to (48x80)
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # Second downsample: 96x160 → 48x80
            norm_layer(128),
            nn.ReLU(inplace=True)
        )

        # ✅ od_N4 (96x160) → (24x40)
        self.conv_od4 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0, bias=False),  # Channel transformation
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # Downsample to (48x80)
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # Further downsample to (24x40)
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 48×80 → 24×40 ✅
            norm_layer(256),
            nn.ReLU(inplace=True)
        )

        # ✅ od_N5 (96x160) → (12x20)
        self.conv_od5 = nn.Sequential(
            nn.Conv2d(192, 512, kernel_size=1, stride=1, padding=0, bias=False),  # Channel transformation
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # Downsample to (48x80)
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # Downsample to (24x40)
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # Downsample to (12x20)
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 24x40 → 12x20 ✅
            norm_layer(512),
            nn.ReLU(inplace=True)
        )
        # Apply pruning
        # self.apply(lambda module: apply_pruning(module, amount=prune_amount))

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()
        # n1, c1, h1, w1 = guide1.size()
        n2, c2, h2, w2 = guide2.size()
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([guide2_down, guide1_down, x], 1)
        f_cat = self.conv_cat(x_cat)
        f_center = self.conv_center(x_cat)
        f_cat = f_cat.view(n, self.out_channels, h * w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)
        f_center_norm = self.norm_center(f_center_norm)
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))

        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)

        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)
        guide_cat_conv_drivable = self.conv_affinity0_drivable(guide_cat)
        guide_cat_conv_lane = self.conv_affinity0_lane(guide_cat)
        guide_cat_conv_object = self.conv_affinity0_object(guide_cat)
        # print("guide_cat_conv_object shape is: ", guide_cat_conv_object.shape)
        guide_cat_value_avg_drivable = guide_cat_conv_drivable + value_avg
        guide_cat_value_avg_lane = guide_cat_conv_lane + value_avg
        guide_cat_value_avg_object = guide_cat_conv_object + value_avg
        # print("guide_cat_value_avg_object shape is: ", guide_cat_value_avg_object.shape)
        f_affinity_drivable = self.conv_affinity1_drivable(guide_cat_value_avg_drivable)
        f_affinity_lane = self.conv_affinity1_lane(guide_cat_value_avg_lane)
        f_affinity_object = self.conv_affinity1_object(guide_cat_value_avg_object)
        # print("f_affinity_object shape is: ", f_affinity_object.shape)
        n_aff, c_ff, h_aff, w_aff = f_affinity_lane.size()
        f_affinity_drivable = f_affinity_drivable.view(n_aff, c_ff, h_aff * w_aff)
        f_affinity_lane = f_affinity_lane.view(n_aff, c_ff, h_aff * w_aff)
        f_affinity_object_reshape = f_affinity_object.view(n_aff, c_ff, h_aff * w_aff)
        # print("f_affinity_object shape is: ", f_affinity_object.shape)
        norm_aff = ((self.center_channels) ** -.5)
        x_up_drivable = norm_aff * x_center.bmm(f_affinity_drivable)
        x_up_lane = norm_aff * x_center.bmm(f_affinity_lane)
        x_up_object = norm_aff * x_center.bmm(f_affinity_object_reshape)
        # print("x_up_object shape is: ", x_up_object.shape)
        x_up_drivable = x_up_drivable.view(n, self.out_channels, h_aff, w_aff)
        x_up_lane = x_up_lane.view(n, self.out_channels, h_aff, w_aff)
        x_up_object = x_up_object.view(n, self.out_channels, h_aff, w_aff)
        # print("x_up_object shape is: ", x_up_object.shape)
        # Free memory before concatenation
        # del guide_cat_value_avg_drivable, guide_cat_value_avg_lane, f_affinity_drivable, f_affinity_lane, x_center
        # torch.cuda.empty_cache()
        x_up_cat_drivable = torch.cat([x_up_drivable, guide_cat_conv_drivable], 1)
        x_up_cat_lane = torch.cat([x_up_lane, guide_cat_conv_lane], 1)
        x_up_cat_object = torch.cat([x_up_object, guide_cat_conv_object], 1)
        # print("x_up_cat_object shape is: ", x_up_cat_object.shape)
        x_up_conv_drivable = self.conv_up_drivable(x_up_cat_drivable)
        x_up_conv_lane = self.conv_up_lane(x_up_cat_lane)
        x_up_conv_object = self.conv_up_object(x_up_cat_object)
        # print("x_up_conv_object shape is: ", x_up_conv_object.shape)
        outputs = (x_up_conv_drivable, x_up_conv_lane)
        # 🟢 Object Detection Feature Maps (OD Head)
        od_N2 = self.conv_od2(x_up_conv_object)  # (96x160x64)
        od_N3 = self.conv_od3(f_affinity_object)  # (48x80x128)
        od_N4 = self.conv_od4(f_affinity_object)  # (24x40x256)
        od_N5 = self.conv_od5(f_affinity_object)  # (12x20x512)
        return outputs, [od_N2, od_N3, od_N4, od_N5]


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

class MultiHGDModule_32(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModule_32, self).__init__()
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
            nn.Conv2d(in_channels, out_channels, 1, bias=False), ######
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
        guide_cat_drivable = x_up0  # torch.cat(x_up0, 1)  # m8_drivable - n x 3*in_channels x h2 x w2 (512x48x80)
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

class MultiHGDModule_lane(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(MultiHGDModule_lane, self).__init__()
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
        self.conv_affinity0 = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
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
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)  # m8 - n x 3*in_channels x h2 x w2 (1536x48x80)
        guide_cat_conv_lane = self.conv_affinity0(guide_cat)  # G_lane - n x out_channels x h2 x w2
        # guide_cat_conv_drivable = self.conv_affinity0(guide_cat)  # G_drivable - n x out_channels x h2 x w2
        guide_cat_value_avg_lane = guide_cat_conv_lane + value_avg  # n x out_channels x h2 x w2
        # guide_cat_value_avg_drivable = guide_cat_conv_drivable + value_avg  # n x out_channels x h2 x w2
        f_affinity_lane = self.conv_affinity1(guide_cat_value_avg_lane)  # W_lane - n x center_channels x h2 x w2
        # f_affinity_drivable = self.conv_affinity1(guide_cat_value_avg_drivable)  # W_drivable - n x center_channels x h2 x w2
        n_aff, c_ff, h_aff, w_aff = f_affinity_lane.size()
        f_affinity_lane = f_affinity_lane.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        # f_affinity_drivable = f_affinity_drivable.view(n_aff, c_ff, h_aff * w_aff)  # # n x center_channels x h2*w2
        norm_aff = ((self.center_channels) ** -.5)
        # x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up_lane = norm_aff * x_center.bmm(f_affinity_lane)  # n x out_channels x h2*w2
        # x_up_drivable = norm_aff * x_center.bmm(f_affinity_drivable)  # n x out_channels x h2*w2
        x_up_lane = x_up_lane.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_lane -  n x out_channels x h2 x w2
        # x_up_drivable = x_up_drivable.view(n, self.out_channels, h_aff, w_aff)  # f_8_tilde_drivable -  n x out_channels x h2 x w2
        x_up_cat_lane = torch.cat([x_up_lane, guide_cat_conv_lane], 1)  # f_8_hat_lane
        # x_up_cat_drivable = torch.cat([x_up_drivable, guide_cat_conv_drivable], 1)  # f_8_hat_drivable
        x_up_conv_lane = self.conv_up(x_up_cat_lane)
        # x_up_conv_drivable = self.conv_up(x_up_cat_drivable)
        # outputs = (x_up_conv_lane, x_up_conv_drivable)
        return x_up_conv_lane#outputs


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
        outs_pred3_drivable = self.conv_pred3_drivable(outs0[0])
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

class MultiHGDecoderTwinLiteNet2(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoderTwinLiteNet2, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(256, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(131, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv30 = nn.Sequential(
           nn.Conv2d(19, 64, 1, padding=0, bias=False),
           norm_layer(64),
           nn.ReLU(inplace=True))
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
        self.hgdmodule0 = MultiHGDModuleTwinLiteNetv2(in_channels=64, center_channels=192, out_channels=32, norm_layer=norm_layer)#, self.num_center,out_channels = 32, norm_layer=norm_layer)
        # self.conv_pred3_lanes = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                       nn.Conv2d(512, out_channels, 1, padding=0))
        # self.conv_pred3_drivable = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                          nn.Conv2d(512, out_channels, 1, padding=0))
        # Adding 1x1 conv to adjust the number of channels
        # feugoun logw redundancy
        # self.conv1x1_1_drivable = nn.Conv2d(32, 16, 1)  # adjust the input and output channels appropriately
        # self.conv1x1_2_drivable = nn.Conv2d(16, 8, 1)  # adjust the input and output channels appropriately
        # self.conv1x1_3_drivable = nn.Conv2d(8, 2, 1)  # adjust the input and output channels appropriately
        #
        # self.conv1x1_1_lanes = nn.Conv2d(32, 16, 1)  # adjust the input and output channels appropriately
        # self.conv1x1_2_lanes = nn.Conv2d(16, 8, 1)  # adjust the input and output channels appropriately
        # self.conv1x1_3_lanes = nn.Conv2d(8, 2, 1)  # adjust the input and output channels appropriately
        # self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, *inputs):

        feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        feat_8 = self.conv30(feat_res8)
        # outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  #
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_8)  #
        # outs_pred3_lanes = self.conv_pred3_lanes(outs0[0])
        # outs_pred3_drivable = self.conv_pred3_drivable(outs0[0])
        # outs_drivable = self.conv1x1_1_drivable(outs0[0])
        # outs_drivable = self.conv1x1_2_drivable(outs_drivable)
        # outs_drivable = self.conv1x1_3_drivable(outs_drivable)
        # outs_lane = self.conv1x1_1_lanes(outs0[1])
        # outs_lane = self.conv1x1_2_lanes(outs_lane)
        # outs_lane = self.conv1x1_3_lanes(outs_lane)
        # outs = (outs_drivable, outs_lane)

        return outs0



class MHGDTwinLiteNet2Scaled(nn.Module):
    def __init__(self, scale, num_center, norm_layer=None, up_kwargs=None):
        super(MHGDTwinLiteNet2Scaled, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(int(256 * scale), int(64 * scale), 1, padding=0, bias=False),
            norm_layer(int(64 * scale)),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(int(131 * scale), int(64 * scale), 1, padding=0, bias=False),
            # nn.Conv2d(int(128 * scale), int(64 * scale), 1, padding=0, bias=False),
            norm_layer(int(64 * scale)),
            nn.ReLU(inplace=True))
        self.conv30 = nn.Sequential(
            nn.Conv2d(int(19 * scale), int(64 * scale), 1, padding=0, bias=False),
            # nn.Conv2d(int(64 * scale), int(64 * scale), 2, padding=0, bias=False),
            norm_layer(int(64 * scale)),
            nn.ReLU(inplace=True))

        self.num_center = int(num_center * scale)
        self.hgdmodule0 = MultiHGDModuleTwinLiteNetv2Scaled(in_channels=int(64 * scale), center_channels=int(192 * scale),
                                                      out_channels=int(32 * scale), norm_layer=norm_layer)

    def forward(self, *inputs):
        feat_res8, feat_res16, feat_res32 = inputs#[0]
        feat_32 = self.conv50(feat_res32)
        feat_16 = self.conv40(feat_res16)
        feat_8 = self.conv30(feat_res8)
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_8)
        return outs0

class MHGDTwinLiteNet2ScaledObjectDetection(nn.Module):
    def __init__(self, in_channels, out_channels,scale, num_center, norm_layer=None, up_kwargs=None):
        super(MHGDTwinLiteNet2ScaledObjectDetection, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(int(256 * scale), int(64 * scale), 1, padding=0, bias=False),
            norm_layer(int(64 * scale)),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(int(131 * scale), int(64 * scale), 1, padding=0, bias=False),
            norm_layer(int(64 * scale)),
            nn.ReLU(inplace=True))
        self.conv30 = nn.Sequential(
            nn.Conv2d(int(19 * scale), int(64 * scale), 1, padding=0, bias=False),
            norm_layer(int(64 * scale)),
            nn.ReLU(inplace=True))

        # ADD NEW CONVOLUTIONS for Object Detection (OD)
        # self.conv_od3 = nn.Sequential(
        #     nn.Conv2d(int(19 * scale), 128, 1, padding=0, bias=False),  # Create N3_OD (45x80, 128)
        #     norm_layer(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_od4 = nn.Sequential(
        #     nn.Conv2d(int(131 * scale), 256, 1, padding=0, bias=False),  # Create N4_OD (24x40, 256)
        #     norm_layer(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2)  # Downsample from (45x80) → (24x40)
        # )
        # self.conv_od5 = nn.Sequential(
        #     nn.Conv2d(int(256 * scale), 512, 1, padding=0, bias=False),  # Create N5_OD (12x20, 512)
        #     norm_layer(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2)  # Downsample from (24x40) → (12x20)
        # )

        self.num_center = int(num_center * scale)
        self.hgdmodule0 = MultiHGDModuleTwinLiteNetv2ScaledObjectDetection(in_channels=int(64 * scale), center_channels=int(192 * scale),
                                                      out_channels=int(32 * scale), norm_layer=norm_layer)

    def forward(self, *inputs):
        feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)
        feat_16 = self.conv40(feat_res16)
        feat_8 = self.conv30(feat_res8)
        outs0, [od_N2, od_N3, od_N4, od_N5] = self.hgdmodule0(feat_32, feat_16, feat_8)
        # 🎯 **New: Object Detection Features**
        # od_N3 = self.conv_od3(feat_res8)  # 45x80 → 128 channels
        # od_N4 = self.conv_od4(feat_res16)  # 24x40 → 256 channels
        # od_N5 = self.conv_od5(feat_res32)  # 12x20 → 512 channels
        return outs0, [od_N2, od_N3, od_N4, od_N5]


class MultiHGDecoderTwinLiteNet2HalfCodewords(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoderTwinLiteNet2HalfCodewords, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(256, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(131, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv30 = nn.Sequential(
           nn.Conv2d(19, 64, 1, padding=0, bias=False),
           norm_layer(64),
           nn.ReLU(inplace=True))
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
        self.hgdmodule0 = MultiHGDModuleTwinLiteNetv2HalfCodewords(in_channels=64, center_channels=192, out_channels=32, norm_layer=norm_layer)#, self.num_center,out_channels = 32, norm_layer=norm_layer)
        # self.conv_pred3_lanes = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                       nn.Conv2d(512, out_channels, 1, padding=0))
        # self.conv_pred3_drivable = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                          nn.Conv2d(512, out_channels, 1, padding=0))
        # Adding 1x1 conv to adjust the number of channels
        # feugoun logw redundancy
        # self.conv1x1_1_drivable = nn.Conv2d(32, 16, 1)  # adjust the input and output channels appropriately
        # self.conv1x1_2_drivable = nn.Conv2d(16, 8, 1)  # adjust the input and output channels appropriately
        # self.conv1x1_3_drivable = nn.Conv2d(8, 2, 1)  # adjust the input and output channels appropriately
        #
        # self.conv1x1_1_lanes = nn.Conv2d(32, 16, 1)  # adjust the input and output channels appropriately
        # self.conv1x1_2_lanes = nn.Conv2d(16, 8, 1)  # adjust the input and output channels appropriately
        # self.conv1x1_3_lanes = nn.Conv2d(8, 2, 1)  # adjust the input and output channels appropriately
        # self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, *inputs):

        feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        feat_8 = self.conv30(feat_res8)
        # outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  #
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_8)  #
        # outs_pred3_lanes = self.conv_pred3_lanes(outs0[0])
        # outs_pred3_drivable = self.conv_pred3_drivable(outs0[0])
        # outs_drivable = self.conv1x1_1_drivable(outs0[0])
        # outs_drivable = self.conv1x1_2_drivable(outs_drivable)
        # outs_drivable = self.conv1x1_3_drivable(outs_drivable)
        # outs_lane = self.conv1x1_1_lanes(outs0[1])
        # outs_lane = self.conv1x1_2_lanes(outs_lane)
        # outs_lane = self.conv1x1_3_lanes(outs_lane)
        # outs = (outs_drivable, outs_lane)

        return outs0

class MultiHGDecoderTwinLiteNet2QuarterCodewords(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoderTwinLiteNet2QuarterCodewords, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(256, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(131, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv30 = nn.Sequential(
           nn.Conv2d(19, 64, 1, padding=0, bias=False),
           norm_layer(64),
           nn.ReLU(inplace=True))
        self.num_center = num_center
        self.hgdmodule0 = MultiHGDModuleTwinLiteNetv2QuarterCodewords(in_channels=64, center_channels=192, out_channels=32, norm_layer=norm_layer)#, self.num_center,out_channels = 32, norm_layer=norm_layer)
    def forward(self, *inputs):

        feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        feat_8 = self.conv30(feat_res8)
        # outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  #
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_8)

        return outs0

class MultiHGDecoderTwinLiteNet2DoubleCodewords(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoderTwinLiteNet2DoubleCodewords, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d  # Default normalization layer
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(256, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(131, 64, 1, padding=0, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True))
        self.conv30 = nn.Sequential(
           nn.Conv2d(19, 64, 1, padding=0, bias=False),
           norm_layer(64),
           nn.ReLU(inplace=True))
        self.num_center = num_center
        self.hgdmodule0 = MultiHGDModuleTwinLiteNetv2DoubleCodewords(in_channels=64, center_channels=192, out_channels=32, norm_layer=norm_layer)#, self.num_center,out_channels = 32, norm_layer=norm_layer)
    def forward(self, *inputs):

        feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        feat_8 = self.conv30(feat_res8)
        # outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  #
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_8)

        return outs0


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

class MultiHGDecoder_32(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoder_32, self).__init__()
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
        self.hgdmodule0 = MultiHGDModule_32(256, self.num_center, 512, norm_layer=norm_layer)
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

class MultiHGDecoder_lane(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(MultiHGDecoder_lane, self).__init__()
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
        self.hgdmodule0 = MultiHGDModule_lane(256, self.num_center, 512, norm_layer=norm_layer)
        # self.conv_pred3_lanes = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                       nn.Conv2d(512, out_channels, 1, padding=0))
        # self.conv_pred3_drivable = nn.Sequential(nn.Dropout2d(0.1, False),
        #                                          nn.Conv2d(512, out_channels, 1, padding=0))
        # Adding 1x1 conv to adjust the number of channels
        self.conv1x1_1_lanes = nn.Conv2d(512, 256, 1)  # adjust the input and output channels appropriately
        self.conv1x1_2_lanes = nn.Conv2d(256, 128, 1)  # adjust the input and output channels appropriately
        self.conv1x1_3_lanes = nn.Conv2d(128, 32, 1)  # adjust the input and output channels appropriately
        # self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)
        # self.conv1x1_1_drivable = nn.Conv2d(512, 256, 1)  # adjust the input and output channels appropriately
        # esvisa to panw gia to lane
        self.conv1x1_2_drivable = nn.Conv2d(256, 128, 1)  # adjust the input and output channels appropriately
        self.conv1x1_3_drivable = nn.Conv2d(128, 32, 1)  # adjust the input and output channels appropriately

    def forward(self, *inputs):

        feat_res2, feat_res4, feat_res8, feat_res16, feat_res32 = inputs[0]
        feat_32 = self.conv50(feat_res32)  # self.conv50(inputs[-1])
        feat_16 = self.conv40(feat_res16)  # self.conv40(inputs[-2])
        outs0 = self.hgdmodule0(feat_32, feat_16, feat_res8)  # 1, 512, 48, 80

        outs_lane = self.conv1x1_1_lanes(outs0)
        outs_lane = self.conv1x1_2_lanes(outs_lane)
        outs_lane = self.conv1x1_3_lanes(outs_lane)
        # outs_drivable = self.conv1x1_1_drivable(feat_res16)
        outs_drivable = self.conv1x1_2_drivable(feat_16)
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
