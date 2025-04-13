import torch
from torch import tensor
import torch.nn as nn
from torch.nn import Conv2d
import sys, os
import math
import sys

sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from lib.models.common_yolopv3 import GhostConv, RepConv, PaFPNELAN_C2, Conv, seg_head, PSA_p
from lib.models.common_yolopv3 import ELANBlock_Head, FPN_C5, FPN_C2, ELANBlock_Head_Ghost, Repconv_Block, ELANNet, \
    PaFPNELAN_Ghost_C2, IDetect, LitePAN
from lib.models.TwinLite_2_scaled_Object_Detection import ESPNet2_Encoder_scaledExtended, UPx2_scaled
# from torchsummary import summary

# TwinLiteNet2Scaled = [
#     [1, 2, 3],  # Det_out_idx, Da_Seg_out_idx, LL_Seg_out_idx
#
#     ###### Backbone
#     [-1, ESPNet2_Encoder_scaled, [5, 3, 1.0]],  # 0 (p=2, q=3, scale=1.0)
#
#     ###### Object Detection Head
#     [ -1, IDetect,  [1, [[4.15629,  11.41984,  5.94761,  16.46950,  8.18673,  23.52688], [12.04416,  29.51737, 16.35089,  41.95507, 24.17928,  57.18741], [33.29597,  78.16243, 47.86408, 108.28889, 36.33312, 189.21414], [73.09806, 144.64581, 101.18080, 253.37000, 136.02821, 408.82248]], [64, 128, 256, 512]]], #3 Detection head
#
#     ###### Drivable Area Segmentation Head
#     [0, UPx2_scaled, [32, 2]],  # 2 (classifier_1: drivable area segmentation)
#
#     ###### Lane Detection Segmentation Head
#     [0, UPx2_scaled, [32, 2]],  # 3 (classifier_2: lane segmentation)
# ]

TwinLiteNet2Scaled = [
    [3, 4, 5],

    # Backbone
    [-1, ESPNet2_Encoder_scaledExtended, [5, 3, 1.0]],

    # Neck edw me to litepan na ksanadw me to -1 h to 0 ti ginetai
    [-1, PaFPNELAN_Ghost_C2, []],  # Output of encoder must be a tuple/list of C3, C4, C5

    # Repconv_Block
    [-1, Repconv_Block, []],

    # Detection Head
    [-1, IDetect, [1, [[4.15629, 11.41984, 5.94761, 16.46950, 8.18673, 23.52688],
                       [12.04416, 29.51737, 16.35089, 41.95507, 24.17928, 57.18741],
                       [33.29597, 78.16243, 47.86408, 108.28889, 36.33312, 189.21414],
                       [73.09806, 144.64581, 101.18080, 253.37000, 136.02821, 408.82248]], [128, 256, 512, 1024]]],

    # Heads
    [0, UPx2_scaled, [32, 2]],
    [0, UPx2_scaled, [32, 2]],
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        # 27
        self.det_out_idx = block_cfg[0][0]

        # 63 67
        self.seg_out_idx = block_cfg[0][1:]

        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is IDetect:
                # detector_index  # 27
                self.detector_index = i

            block_ = block(*args)

            block_.index, block_.from_ = i, from_

            layers.append(block_)

            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, IDetect):
            s = 512  # 2x min stride auto thelei psaksimoooooo. tha mporouse na ginei meleti edwwwwww
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                # detects = model_out[0]
                detects = model_out[0] if isinstance(model_out[0], list) else model_out[0][1]
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward

            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        seg_feats = None  # da_feat, ll_feat

        for i, block in enumerate(self.model):
            x = cache[block.from_] if block.from_ != -1 else x

            # Dispatch: Only handle complex blocks separately
            if isinstance(block, PaFPNELAN_Ghost_C2):
                x = block(x[:4]) if isinstance(x, (list, tuple)) else block(x)
            elif isinstance(block, IDetect):
                x = block(x)
                det_out = x
            elif i in self.seg_out_idx:
                seg_index = self.seg_out_idx.index(i)
                x = block(x[4:][seg_index])
            else:
                x = block(x)

            if i in self.seg_out_idx:
                out.append(x)
            cache.append(x if block.index in self.save else None)

        out.insert(0, det_out)
        return out


    # def forward(self, x):
    #     cache = []
    #     out = []
    #     det_out = None
    #     Da_fmap = []
    #     LL_fmap = []
    #     for i, block in enumerate(self.model):
    #         if block.from_ != -1:
    #             x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
    #                                                                          block.from_]  # calculate concat detect
    #         #print("i is: ", i)
    #         # x = block(x)
    #         if isinstance(block, IDetect):
    #             # 💡 Print feature map sizes for the detection head
    #             # print(f"\n🚀 Input to IDetect (Layer {i}):")
    #             # if isinstance(x, tuple):
    #             #     feature_maps = x[1]  # Extract list of feature maps
    #             #     print(f"\n🚀 Input Feature Maps to IDetect (Layer {i}):")
    #             #     for j, fmap in enumerate(feature_maps):
    #             #         print(f"  Feature map {j}: {fmap.shape}")
    #             # If block is IDetect, extract the list of feature maps from the tuple
    #             x = block(x[1] if isinstance(x, tuple) else x)
    #
    #
    #
    #         # else:
    #         #     # For other blocks, use the first element of the tuple or the tensor itself
    #         #     x = block(x[0] if isinstance(x, tuple) else x)
    #         elif i in self.seg_out_idx:
    #             # Handle segmentation heads (UPx2_scaled), use the specific tensor from the tuple
    #             if isinstance(x, tuple) and len(x) == 2:
    #                 # Choose the correct tensor for each segmentation head
    #                 seg_index = self.seg_out_idx.index(i)  # Determine which segmentation it is
    #                 x = block(x[0][seg_index])
    #             else:
    #                 x = block(x)
    #         else:
    #             # General case for other layers
    #             x = block(x[0] if isinstance(x, tuple) else x)
    #
    #         if i in self.seg_out_idx:  # save driving area segment result
    #             out.append(x)
    #         if i == self.detector_index:
    #             det_out = x
    #         cache.append(x if block.index in self.save else None)
    #     out.insert(0, det_out)
    #     return out

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, IDetect):
                m.fuse()
                m.forward = m.fuseforward
        return self

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

# class MCnet(nn.Module):
#     def __init__(self, block_cfg, **kwargs):
#         super(MCnet, self).__init__()
#         layers, save = [], []
#         self.nc = 1  # Number of classes for detection
#         self.detector_index = -1
#
#         # Output indices for detection, drivable area segmentation, and lane segmentation
#         self.det_out_idx = block_cfg[0][0]  # Index of detection output
#         self.seg_out_idx = block_cfg[0][1:]  # Indices of segmentation outputs
#
#         # Build layers
#         for i, (from_, block, args) in enumerate(block_cfg[1:]):
#             block = eval(block) if isinstance(block, str) else block  # eval strings
#             if block is IDetect:
#                 self.detector_index = i  # Set detector index
#
#             block_ = block(*args)
#             block_.index, block_.from_ = i, from_
#             layers.append(block_)
#             save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
#
#         self.model, self.save = nn.Sequential(*layers), sorted(save)
#         self.names = [str(i) for i in range(self.nc)]
#
#         # Initialize strides and anchors for the detection head
#         if self.detector_index != -1:
#             self._initialize_detector()
#
#         # Initialize weights
#         initialize_weights(self)
#
#     def _initialize_detector(self):
#         """Initialize strides and anchors for the detection head."""
#         Detector = self.model[self.detector_index]  # Get the detection head
#         if isinstance(Detector, IDetect):
#             s = 512  # 2x min stride
#             with torch.no_grad():
#                 # Forward pass with a dummy input to initialize strides and anchors
#                 model_out = self.forward(torch.zeros(1, 3, s, s))
#                 detects = model_out[0]
#                 Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # Set strides
#
#             # Adjust anchors for the corresponding scale
#             Detector.anchors /= Detector.stride.view(-1, 1, 1)
#             check_anchor_order(Detector)  # Ensure anchors are in the correct order
#             self.stride = Detector.stride
#
#             # Initialize biases
#             self._initialize_biases()
#
#     def _initialize_biases(self, cf=None):
#         """Initialize biases for the detection head."""
#         m = self.model[self.detector_index]
#         for mi, s in zip(m.m, m.stride):
#             b = mi.bias.view(m.na, -1)
#             b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#             b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
#             mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#
#     def forward(self, x):
#         # Backbone forward pass
#         backbone_output = self.model[0](x)  # ESPNet2_Encoder_scaled
#
#         # Unpack backbone outputs
#         segmentation_heads, od_head = backbone_output
#         da_feature, lane_feature = segmentation_heads  # Unpack segmentation features
#
#         # Initialize cache and outputs
#         cache = []
#         out = []
#         det_out = None
#
#         # Process layers
#         for i, block in enumerate(self.model[1:]):  # Skip backbone (already processed)
#             if block.from_ != -1:
#                 # Route inputs based on block.from_
#                 if i == self.detector_index:
#                     # Feed od_head to IDetect
#                     x = od_head
#                 elif i == self.seg_out_idx[0]:
#                     # Feed da_feature to drivable area segmentation head
#                     x = da_feature
#                 elif i == self.seg_out_idx[1]:
#                     # Feed lane_feature to lane detection head
#                     x = lane_feature
#                 else:
#                     # Default behavior (e.g., for neck layers)
#                     x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
#                                                                                  block.from_]
#
#             # Forward pass through the block
#             x = block(x)
#
#             # Save outputs
#             if i == self.detector_index:
#                 det_out = x  # Save detection output
#             elif i in self.seg_out_idx:
#                 out.append(x)  # Save segmentation outputs
#
#             # Update cache
#             cache.append(x if block.index in self.save else None)
#
#         # Insert detection output at the beginning
#         out.insert(0, det_out)
#         return out
def get_net(cfg, **kwargs):
    m_block_cfg = TwinLiteNet2Scaled
    model = MCnet(m_block_cfg, **kwargs)
    return model


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

# from torchviz import make_dot


if __name__ == "__main__":
    model = get_net(TwinLiteNet2Scaled)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(1, 3, 384, 640).to(device)  # Dummy input
    model_out = model(x)
    # make_dot(model_out, params=dict(model.named_parameters())).render("model_graph", format="png")

    # input_size = (3, 512, 512)  # Adjust based on your dataset
    # Print summary
    # summary(model, input_size=input_size)

    encoder = ESPNet2_Encoder_scaledExtended(2, 3, 1.0).to(device)  # Example

    # eisodos sto object detection head tou yolopv3:
    # N2, N3, N4, N5 = 96x160x64, 48x80x128, 24x40x256, 12x20x512 (Height x Width x Channels)
    # ara na vrw ta antistoixa
    # x = torch.randn(1, 3, 360, 640)  # Example input
    output = encoder(x)

    backbone = ESPNet2_Encoder_scaledExtended(2, 3, 1.0).to(device)
    output = backbone(x)

    # Print feature map sizes
    for i, feat in enumerate(output):
        print(f"Feature C{i} shape: {len(feat)}")


    # Test with dummy input
    input_tensor = torch.randn(1, 3, 360, 640)
    # output = model(input_tensor)

    # Output shapes
    det_out = output[0]
    da_out = output[1]
    ll_out = output[2]

    print(f"Detection output shape: {det_out}")
    print(f"Drivable area output shape: {da_out.shape}")
    print(f"Lane line output shape: {ll_out.shape}")

    pass
    # from torch.utils.tensorboard import SummaryWriter
    # model = get_net(False)
    # input_ = torch.randn((1, 3, 256, 256))
    # gt_ = torch.rand((1, 2, 256, 256))
    # metric = SegmentationMetric(2)
    # model_out,SAD_out = model(input_)
    # detects, dring_area_seg, lane_line_seg = model_out
    # Da_fmap, LL_fmap = SAD_out
    # for det in detects:
    #     print(det.shape)
    # print(dring_area_seg.shape)
    # print(lane_line_seg.shape)
