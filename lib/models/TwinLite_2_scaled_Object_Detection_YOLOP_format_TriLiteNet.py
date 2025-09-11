import torch
from torch import tensor
import torch.nn as nn
from torch.nn import Conv2d
import sys, os
import math
import sys

sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from lib.utils import check_anchor_order
from lib.models.common_yolopv3 import RepConv, Conv
from lib.models.common_yolopv3 import IDetect
from lib.models.TwinLite_2_scaled_Object_Detection import ESPNet2_Encoder_scaledExtended, MHGDTwinLiteNet2Scaled, UPx2_scaled, ESPNet2_Encoder_scaledExtendedDWS
from lib.models.common_trilitenet import Encoder, DetectHead, sc_ch_dict, Detect
# from torchsummary import summary
import copy
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class SegProtoHead(nn.Module):
    """
    YOLO-style prototype mask head — παίρνει P3 και βγάζει P masks (π.χ. P=32)
    """
    def __init__(self, in_ch: int, num_proto: int = 32):
        super().__init__()
        self.num_proto = num_proto
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_proto, 1)  # τελικό output: [B, P, H, W]
        )

    def forward(self, x):
        """
        x: P3 feature map (από DetectHead) — shape [B, C, H, W]
        returns: [B, P, H, W]
        """
        return self.block(x)

class DetectSeg(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=2, anchors=(), ch=(), num_proto=32):  # detection + mask coefficients
        super().__init__()
        self.nc = nc
        self.num_proto = num_proto
        self.no = nc + 5 + num_proto  # cls + box + obj + coeffs
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid

        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []  # inference output
        outputs = []  # raw outputs (for loss)
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny * nx).permute(0, 1, 3, 2).view(
                bs, self.na, ny, nx, self.no
            ).contiguous()

            outputs.append(x[i])  # keep raw outputs for loss

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()

                # xywh decode
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # x, y
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # w, h

                z.append(y.view(bs, -1, self.no))  # flatten all anchors

        if self.training:
            return outputs  # raw format for loss
        else:
            return (torch.cat(z, 1), outputs)  # predictions + raw outputs

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')  # PyTorch >=1.10
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Proto_yolov12(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        # sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class Detect_yolov12(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Segment_yolov12(Detect_yolov12):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Proto(nn.Module):
    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)  # final: (B, 32, H, W)
    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Segment(nn.Module):
    stride = None  # Set later based on input/output scale

    def __init__(self, nc=3, nm=32, npr=39, ch=()):  # ch = [128, 256, 512]
        super().__init__()
        self.nc = nc
        self.nm = nm
        self.npr = npr
        self.no = 5 + nc + nm
        self.nl = len(ch)  # number of feature maps (3)
        self.na = 3  # anchors per location (YOLO-style)

        # Create conv layers to predict outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

        # Prototype mask module
        self.proto = Proto(c1=ch[0], c_=256, c2=npr)

        # Grids and anchors
        self.grid = [torch.zeros(1)] * self.nl
        anchors = [[4, 12, 7, 19, 11, 28], [17, 40, 25, 58, 38, 89], [62, 136, 88, 206, 124, 412]]
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))

    def forward(self, x):  # x = [P3, P4, P5]
        z = []
        bs = x[0].shape[0]
        mask_protos = self.proto(x[0])  # (B, npr, H, W)

        for i in range(self.nl):
            xi = self.m[i](x[i])  # Conv2d → (B, no * na, H, W)
            bs, _, ny, nx = xi.shape
            xi = xi.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != xi.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(xi.device)

                y = xi.clone().sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))

            x[i] = xi

        if self.training:
            return x, mask_protos
        else:
            return torch.cat(z, 1), mask_protos

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



def TwinLiteNet2Scaled(model_cfg):
    print("[DEBUG] model_cfg passed to Segment:", model_cfg)

    TwinLiteNet2Scaled = [
        [2, [4]],

        # Backbone
        [-1, Encoder, [model_cfg]],

        # Neck edw me to litepan na ksanadw me to -1 h to 0 ti ginetai
        [-1, DetectHead, [model_cfg]],  # Output of encoder must be a tuple/list of C3, C4, C5

        # Detection Head
        [-1, Segment, [3,  # number of instance classes
            model_cfg.get('nm', 32),  # number of mask coefficients
            model_cfg.get('npr', 39),  # number of prototype masks
            [model_cfg['chanels'][3]] * 3  # channels for P3, P4, P5
        ]],
        # DA & LLS Heads
        [0, MHGDTwinLiteNet2Scaled, [1, 64]],
        [-1, UPx2_scaled, [32, 3]]
    ]
    return TwinLiteNet2Scaled


class MCnet(nn.Module):
    def __init__(self, block_cfg,**kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 3
        self.detector_index = -1
        # 27
        self.det_out_idx = block_cfg[0][0]

        # 63 67
        self.seg_out_idx = block_cfg[0][1]
        if isinstance(self.seg_out_idx, int):
            self.seg_out_idx = [self.seg_out_idx]

        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            # if block is IDetect:
            if block is Segment:
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
        # if isinstance(Detector, IDetect):
        if isinstance(Detector, Segment):
            s = 128  # 2x min stride auto thelei psaksimoooooo. tha mporouse na ginei meleti edwwwwww
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 5, s, s))
                detects = model_out[0][0]
                # detects = model_out[0] if isinstance(model_out[0], list) else model_out[0][1]
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
        det_feats, seg_feats = None, None  # Initialize feature holders

        # Forward through each block in the config
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                             block.from_]

            # Handle backbone output (now returns tuple of tuples)
            if i == 0:  # Assuming index 0 is ESPNet2_Encoder_scaledExtended
                det_feats, seg_feats = block(x)  # Unpack (C3-C6), (C2-C4)
                x = det_feats  # For compatibility with subsequent blocks
                cache.append(x)  # Store backbone output in cache
                continue

            # Process detection path (PaFPN -> Repconv -> IDetect)
            elif isinstance(block, DetectHead):
                x = block(det_feats)  # Process C3-C6 features

            # elif isinstance(block, Repconv_Block) or isinstance(block, IDetect):
            elif isinstance(block, Segment):
                # print(f"[DEBUG] Segment input: {[f.shape for f in x]}")
                x = block(x)
                det_out = x

            # Process segmentation path (MHGD)
            elif isinstance(block, MHGDTwinLiteNet2Scaled):
                # x = block(*seg_feats)  # Unpack C2-C4 features
                mhgd_out = block(*seg_feats)
                x = mhgd_out  # To allow caching
            elif i in self.seg_out_idx:
                # seg_index = self.seg_out_idx.index(i)
                if mhgd_out is not None:
                    x = block(mhgd_out)
                else:
                    raise RuntimeError("MHGD output missing at segmentation head")
                out.append(x)
                # if isinstance(x, torch.Tensor):
                #     print(f"[DEBUG] Lane output shape: {x.shape}")  # Should be [B, C, H, W]

            else:
                x = block(x)

            cache.append(x if i in self.save else None)

        out.insert(0, det_out)  # [det_out, da_seg_out, ll_seg_out]
        return out


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

def get_net(cfg, **kwargs):
    model_cfg = sc_ch_dict['base']
    m_block_cfg = TwinLiteNet2Scaled(model_cfg)
    model = MCnet(m_block_cfg, **kwargs)
    # print("[DEBUG] Segment head configuration:")
    # print("  nc =", model.model[model.detector_index].nc)
    # print("  nm =", model.model[model.detector_index].nm)
    # print("  no =", model.model[model.detector_index].no)
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
