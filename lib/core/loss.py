import torch.nn as nn
import torch
from .general import bbox_iou
from .postprocess import build_targets
from lib.core.evaluate import SegmentationMetric
import math

def debug_loss_component(name, value):
    if torch.isnan(value) or torch.isinf(value):
        print(f"🚨 {name} is invalid: {value.item()}")
    else:
        print(f"{name} = {value.item():.6f}")

# class MultiHeadLoss(nn.Module):
#     """
#     collect all the loss we need
#     """
#     def __init__(self, losses, cfg, lambdas=None):
#         """
#         Inputs:
#         - losses: (list)[nn.Module, nn.Module, ...]
#         - cfg: config object
#         - lambdas: (list) + IoU loss, weight for each loss
#         """
#         super().__init__()
#         # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
#         if not lambdas:
#             lambdas = [1.0 for _ in range(len(losses) + 3)]
#         assert all(lam >= 0.0 for lam in lambdas)
#
#         self.losses = nn.ModuleList(losses)
#         self.lambdas = lambdas
#         self.cfg = cfg
#
#     def forward(self, head_fields, head_targets, shapes, model):
#         """
#         Inputs:
#         - head_fields: (list) output from each task head
#         - head_targets: (list) ground-truth for each task head
#         - model:
#
#         Returns:
#         - total_loss: sum of all the loss
#         - head_losses: (tuple) contain all loss[loss1, loss2, ...]
#
#         """
#         # head_losses = [ll
#         #                 for l, f, t in zip(self.losses, head_fields, head_targets)
#         #                 for ll in l(f, t)]
#         #
#         # assert len(self.lambdas) == len(head_losses)
#         # loss_values = [lam * l
#         #                for lam, l in zip(self.lambdas, head_losses)
#         #                if l is not None]
#         # total_loss = sum(loss_values) if loss_values else None
#         # print(model.nc)
#         total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)
#
#         return total_loss, head_losses
#
#     def _forward_impl(self, predictions, targets, shapes, model):
#         """
#
#         Args:
#             predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
#             targets: gts [det_targets, segment_targets, lane_targets]
#             model:
#
#         Returns:
#             total_loss: sum of all the loss
#             head_losses: list containing losses
#
#         """
#         cfg = self.cfg
#         device = targets[0].device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets
#
#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         cp, cn = smooth_BCE(eps=0.0)
#
#         BCEcls, BCEobj, BCEseg = self.losses
#
#         # Calculate Losses
#         nt = 0  # number of targets
#         no = len(predictions[0])  # number of outputs
#         balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
#
#         # calculate detection loss
#         for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
#
#             n = b.shape[0]  # number of targets
#             if n:
#                 nt += n  # cumulative targets
#                 ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
#
#                 # Regression
#                 pxy = ps[:, :2].sigmoid() * 2. - 0.5
#                 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
#                 iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
#                 lbox += (1.0 - iou).mean()  # iou loss
#
#                 # Objectness
#                 tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
#
#                 # Classification
#                 # print(model.nc)
#                 if model.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
#                     t[range(n), tcls[i]] = cp
#                     lcls += BCEcls(ps[:, 5:], t)  # BCE
#             lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
#
#         drive_area_seg_predicts = predictions[1].view(-1)
#         drive_area_seg_targets = targets[1].view(-1)
#         lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)
#
#         lane_line_seg_predicts = predictions[2].view(-1)
#         lane_line_seg_targets = targets[2].view(-1)
#         lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)
#
#         metric = SegmentationMetric(2)
#         nb, _, height, width = targets[1].shape
#         pad_w, pad_h = shapes[0][1][1]
#         pad_w = int(pad_w)
#         pad_h = int(pad_h)
#         _,lane_line_pred=torch.max(predictions[2], 1)
#         _,lane_line_gt=torch.max(targets[2], 1)
#         lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
#         lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
#         metric.reset()
#         metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
#         IoU = metric.IntersectionOverUnion()
#         liou_ll = 1 - IoU
#
#         s = 3 / no  # output count scaling
#         lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
#         lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
#         lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]
#
#         lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
#         lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
#         liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]
#
#
#         if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:
#             lseg_da = 0 * lseg_da
#             lseg_ll = 0 * lseg_ll
#             liou_ll = 0 * liou_ll
#
#         if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#
#         if cfg.TRAIN.LANE_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#             lseg_da = 0 * lseg_da
#
#         if cfg.TRAIN.DRIVABLE_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#             lseg_ll = 0 * lseg_ll
#             liou_ll = 0 * liou_ll
#
#         loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
#         # loss = lseg
#         # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
#         return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), liou_ll.item(), loss.item())

'''edw i teleutaia class gia to loss'''

# class MultiHeadLoss(nn.Module):
#     """
#     Συγκεντρώνει όλους τους επιμέρους loss (det, segmentation, lane segmentation κλπ)
#     """
#     def __init__(self, losses, cfg, lambdas=None):
#         """
#         Args:
#         - losses: λίστα από loss functions [BCEcls, BCEobj, BCEseg, CElane]
#         - cfg: config αρχείο
#         - lambdas: βάρη για κάθε loss
#                   (π.χ. [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] για [lcls, lobj, lbox, lseg_da, lseg_ll, liou_ll])
#         """
#         super().__init__()
#         if not lambdas:
#             lambdas = [1.0 for _ in range(len(losses) + 3)]  # +3 για bbox, iou, liou
#         assert all(lam >= 0.0 for lam in lambdas)
#
#         self.losses = nn.ModuleList(losses)  # [BCEcls, BCEobj, BCEseg, CElane]
#         self.lambdas = lambdas
#         self.cfg = cfg
#
#     def forward(self, head_fields, head_targets, shapes, model):
#         """
#         Args:
#         - head_fields: output από το model (detections + segmentations)
#         - head_targets: ground truth [det, drivable_area, lane_line]
#         """
#         return self._forward_impl(head_fields, head_targets, shapes, model)
#
#     def _forward_impl(self, predictions, targets, shapes, model):
#         cfg = self.cfg
#         device = targets[0].device
#
#         # init detection losses
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#
#         # build YOLO targets
#         tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)
#
#         cp, cn = smooth_BCE(eps=0.0)
#         BCEcls, BCEobj, BCEseg, CElane = self.losses  # ✨ CrossEntropyLoss για lane
#
#         nt = 0
#         no = len(predictions[0])  # #YOLO heads
#         balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]
#
#         # --- Detection loss ---
#         for i, pi in enumerate(predictions[0]):
#             b, a, gj, gi = indices[i]
#             tobj = torch.zeros_like(pi[..., 0], device=device)
#
#             n = b.shape[0]
#             if n:
#                 nt += n
#                 ps = pi[b, a, gj, gi]
#
#                 pxy = ps[:, :2].sigmoid() * 2. - 0.5
#                 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1).to(device)
#
#                 iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
#                 lbox += (1.0 - iou).mean()
#
#                 tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)
#
#                 if model.nc > 1:
#                     t = torch.full_like(ps[:, 5:], cn, device=device)
#                     t[range(n), tcls[i]] = cp
#                     lcls += BCEcls(ps[:, 5:], t)
#
#             lobj += BCEobj(pi[..., 4], tobj) * balance[i]
#
#         # --- Drivable area segmentation loss ---
#         drive_area_seg_predicts = predictions[1].view(-1)
#         drive_area_seg_targets = targets[1].view(-1)
#         lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)
#
#         # --- Lane line segmentation loss (CrossEntropy) ---
#         lane_logits = predictions[2]  # [B, 2, H, W]
#         lane_gt = targets[2].long()   # [B, H, W] με 0 ή 1
#
#         if lane_gt.ndim == 4 and lane_gt.shape[1] == 2:
#             lane_gt = torch.argmax(lane_gt, dim=1)  # Convert one-hot to class indices
#
#         lane_gt = lane_gt.long()
#         # Clamp logits just in case (safe)
#         lane_logits = torch.clamp(lane_logits, min=-30, max=30)
#
#         # Replace NaNs or inf in logits (if any)
#         lane_logits = torch.nan_to_num(lane_logits, nan=0.0, posinf=30.0, neginf=-30.0)
#
#         # Check for invalid labels
#         if torch.any((lane_gt < 0) | (lane_gt >= 2)):
#             print(f"[ERROR] Invalid lane_gt values: {torch.unique(lane_gt)}")
#             lane_gt = torch.clamp(lane_gt, 0, 1)
#         lseg_ll = CElane(lane_logits, lane_gt)  # ✅ το σωστό loss για multi-class logits
#         assert not torch.isnan(lseg_ll).any(), "CrossEntropyLoss returned NaN"
#
#         # --- Lane segmentation IoU για μετρική ---
#         _, lane_pred = torch.max(lane_logits, dim=1)  # [B, H, W]
#         metric = SegmentationMetric(2)
#         nb, _, height, width = targets[1].shape
#         pad_w, pad_h = shapes[0][1][1]
#         pad_w, pad_h = int(pad_w), int(pad_h)
#
#         lane_pred = lane_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
#         lane_gt_crop = lane_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
#
#         metric.reset()
#         metric.addBatch(lane_pred.cpu(), lane_gt_crop.cpu())
#         IoU = metric.IntersectionOverUnion()
#         liou_ll = 1 - IoU  # turn IoU into a loss
#         # print("[DEBUG] lane_logits shape:", lane_logits.shape)  # Π.χ. [B, 2, H, W]
#         # print("[DEBUG] lane_gt shape:", lane_gt.shape)          # Π.χ. [B, H, W]
#         # print("[DEBUG] lane_gt dtype:", lane_gt.dtype)
#         # print("[DEBUG] lane_gt unique values:", torch.unique(lane_gt))
#         # print("[DEBUG] lane_logits min/max:", lane_logits.min().item(), lane_logits.max().item())
#         # --- Apply loss weights ---
#         s = 3 / no
#         lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
#         lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
#         lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]
#
#         lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
#         lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
#         liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]
#
#         # --- Selective training switches ---
#         if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY:
#             lseg_da = 0 * lseg_da
#             lseg_ll = 0 * lseg_ll
#             liou_ll = 0 * liou_ll
#
#         if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#
#         if cfg.TRAIN.LANE_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#             lseg_da = 0 * lseg_da
#
#         if cfg.TRAIN.DRIVABLE_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#             lseg_ll = 0 * lseg_ll
#             liou_ll = 0 * liou_ll
#         print(f"[LOSS DEBUG] lbox: {lbox.item():.4f}, lobj: {lobj.item():.4f}, lcls: {lcls.item():.4f}")
#         print(f"[LOSS DEBUG] lseg_da: {lseg_da.item():.4f}, lseg_ll: {lseg_ll.item():.4f}, liou_ll: {liou_ll:.4f}")
#
#         total_loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
#         assert not torch.isnan(lbox).any(), "NaN detected in lbox"
#         assert not torch.isnan(lobj).any(), "NaN detected in lobj"
#         assert not torch.isnan(lcls).any(), "NaN detected in lcls"
#         assert not torch.isnan(lseg_da).any(), "NaN detected in lseg_da"
#         assert not torch.isnan(lseg_ll).any(), "NaN detected in lseg_ll"
#         assert not math.isnan(liou_ll), "NaN detected in liou_ll"
#         return total_loss, (
#             lbox.item(), lobj.item(), lcls.item(), lseg_da.item(),
#             lseg_ll.item(), liou_ll.item(), total_loss.item()
#         )


# class MultiHeadLoss(nn.Module):
#     """
#     collect all the loss we need
#     """
#
#     def __init__(self, losses, cfg, lambdas=None):
#         """
#         Inputs:
#         - losses: (list)[nn.Module, nn.Module, ...]
#         - cfg: config object
#         - lambdas: (list) + IoU loss, weight for each loss
#         """
#         super().__init__()
#         # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
#         if not lambdas:
#             lambdas = [1.0 for _ in range(len(losses) + 3)]
#         assert all(lam >= 0.0 for lam in lambdas)
#
#         self.losses = nn.ModuleList(losses)
#         self.lambdas = lambdas
#         self.cfg = cfg
#
#     def forward(self, head_fields, head_targets, shapes, model):
#         """
#         Inputs:
#         - head_fields: (list) output from each task head
#         - head_targets: (list) ground-truth for each task head
#         - model:
#
#         Returns:
#         - total_loss: sum of all the loss
#         - head_losses: (tuple) contain all loss[loss1, loss2, ...]
#
#         """
#         # head_losses = [ll
#         #                 for l, f, t in zip(self.losses, head_fields, head_targets)
#         #                 for ll in l(f, t)]
#         #
#         # assert len(self.lambdas) == len(head_losses)
#         # loss_values = [lam * l
#         #                for lam, l in zip(self.lambdas, head_losses)
#         #                if l is not None]
#         # total_loss = sum(loss_values) if loss_values else None
#         # print(model.nc)
#         total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)
#
#         return total_loss, head_losses
#
#     def _forward_impl(self, predictions, targets, shapes, model):
#         """
#
#         Args:
#             predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
#             targets: gts [det_targets, segment_targets, lane_targets]
#             model:
#
#         Returns:
#             total_loss: sum of all the loss
#             head_losses: list containing losses
#
#         """
#         cfg = self.cfg
#         device = targets[0].device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets
#
#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         cp, cn = smooth_BCE(eps=0.0)
#
#         BCEcls, BCEobj, BCEseg = self.losses
#
#         # Calculate Losses
#         nt = 0  # number of targets
#         no = len(predictions[0])  # number of outputs
#         balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
#
#         # calculate detection loss
#         for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
#
#             n = b.shape[0]  # number of targets
#             if n:
#                 nt += n  # cumulative targets
#                 ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
#
#                 # Regression
#                 pxy = ps[:, :2].sigmoid() * 2. - 0.5
#                 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
#                 iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
#                 lbox += (1.0 - iou).mean()  # iou loss
#
#                 # Objectness
#                 tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
#
#                 # Classification
#                 # print(model.nc)
#                 if model.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
#                     t[range(n), tcls[i]] = cp
#                     lcls += BCEcls(ps[:, 5:], t)  # BCE
#             lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
#
#         drive_area_seg_predicts = predictions[1].view(-1)
#         drive_area_seg_targets = targets[1].view(-1)
#         lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)
#
#         # lane_line_seg_predicts = predictions[2].view(-1)
#         # print("Lane line output shape:", predictions[2].shape)
#         # _, lane_line_pred = torch.max(predictions[2], dim=1)
#         # print("Lane line prediction unique values:", torch.unique(lane_line_pred))
#         # print("Lane logits min/max:", predictions[2].min().item(), predictions[2].max().item())
#
#         # lane_line_seg_targets = targets[2].view(-1)
#         # print("Lane line GT sum:", targets[2].sum().item())
#         # lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)
#         # lane_logits = lane_line_seg_predicts  # (B, 2, H, W)
#         # lane_gt = targets[2]  # (B, 2, H, W) ή (B, H, W)
#         CElane = self.losses[3]  # CrossEntropyLoss
#         lane_logits = predictions[2]  # [B, 2, H, W]
#         lane_gt = targets[2].long()  # [B, H, W], class labels 0 or 1
#
#         lseg_ll = CElane(lane_logits, lane_gt)
#         # print("Logits min/max", lane_logits.min().item(), lane_logits.max().item())
#         # print("GT unique", torch.unique(lane_gt))
#         metric = SegmentationMetric(2)
#         nb, _, height, width = targets[1].shape
#         pad_w, pad_h = shapes[0][1][1]
#         pad_w = int(pad_w)
#         pad_h = int(pad_h)
#         _, lane_line_pred = torch.max(predictions[2], 1)
#         _, lane_line_gt = torch.max(targets[2], 1)
#         lane_line_pred = lane_line_pred[:, pad_h:height - pad_h, pad_w:width - pad_w]
#         lane_line_gt = lane_line_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]
#         # total = (lane_line_pred == lane_line_gt).numel()
#         # correct = (lane_line_pred == lane_line_gt).sum()
#         # print("Manual pixel acc:", correct.item() / total)
#         # print("lane_line_pred dtype:", lane_line_pred.dtype)
#         # print("lane_line_gt dtype:", lane_line_gt.dtype)
#         # print("Lane GT after argmax shape:", lane_line_gt.shape)
#         # print("Lane GT unique:", torch.unique(lane_line_gt))
#         # print("pad_h", pad_h, "height", height)
#         # print("pad_w", pad_w, "width", width)
#         # print("lane_line_gt cropped shape:", lane_line_gt.shape)
#         # print("lane_line_pred cropped shape:", lane_line_pred.shape)
#         metric.reset()
#         metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
#         IoU = metric.IntersectionOverUnion()
#         liou_ll = 1 - IoU
#         # liou_ll = torch.tensor(liou_ll, device=device)
#
#         s = 3 / no  # output count scaling
#         lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
#         lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
#         lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]
#
#         lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
#         lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
#         liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]
#
#         if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:
#             lseg_da = 0 * lseg_da
#             lseg_ll = 0 * lseg_ll
#             liou_ll = 0 * liou_ll
#
#         if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#
#         if cfg.TRAIN.LANE_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#             lseg_da = 0 * lseg_da
#
#         if cfg.TRAIN.DRIVABLE_ONLY:
#             lcls = 0 * lcls
#             lobj = 0 * lobj
#             lbox = 0 * lbox
#             lseg_ll = 0 * lseg_ll
#             liou_ll = 0 * liou_ll
#
#         loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
#
#         if torch.isnan(lseg_ll):
#             print("⚠️ lseg_ll is NaN!")
#         # print("\n🔍 DEBUGGING LOSS COMPONENTS:")
#         # debug_loss_component("lbox", lbox)
#         # debug_loss_component("lobj", lobj)
#         # debug_loss_component("lcls", lcls)
#         # debug_loss_component("lseg_da", lseg_da)
#         # debug_loss_component("lseg_ll", lseg_ll)
#         # debug_loss_component("liou_ll", liou_ll)
#         # debug_loss_component("total", loss)
#         # print("🔚 END DEBUG\n")
#         # loss = lseg
#         # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
#         return loss, (
#         lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), liou_ll.item(), loss.item())

import torch
import torch.nn as nn

class MultiHeadLoss(nn.Module):
    """
    Multi-task loss supporting instance segmentation and semantic segmentation.
    """
    def __init__(self, losses, cfg, lambdas=None):
        super().__init__()
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses))]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)
        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """
        predictions: [ (det_preds, proto_masks), semantic_segmentation_output ]
        targets: [ detection_targets (Tensor[N, 6]), semantic_segmentation_target ]
                 detection_targets = (image_idx, class, x1, y1, x2, y2)
        """
        device = predictions[1].device
        cfg = self.cfg

        # ========================
        # 1. Instance Segmentation
        # ========================

        det_preds, proto_masks = predictions[0]
        print(f"det_preds.shape: {det_preds[0].shape}")

        det_pred = det_preds[0]

        # Handle various det_pred shapes
        if det_pred.ndim == 5:
            B, A, H, W, C = det_pred.shape
            det_pred = det_pred.view(B, A * H * W, C)
        elif det_pred.ndim == 4 and det_pred.shape[1] > 5:
            B, C, H, W = det_pred.shape
            det_pred = det_pred.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif det_pred.ndim != 3:
            raise ValueError(f"Unsupported det_pred shape: {det_pred.shape}")
        B, N, C = det_pred.shape

        nm = proto_masks.shape[1]  # number of mask coeffs
        num_classes = C - nm
        print("num_classes =", num_classes)

        mask_coeffs = det_pred[:, :, :nm]
        class_scores = det_pred[:, :, nm:]

        # ========================
        # Convert YOLO Targets → Masks
        # ========================
        det_targets = targets[0].to(device)  # shape (N_total, 6)
        image_indices = det_targets[:, 0].long()  # [N_total]
        class_ids = det_targets[:, 1].long()  # [N_total]
        boxes = det_targets[:, 2:]  # [x1, y1, x2, y2], shape (N_total, 4)

        Hm, Wm = proto_masks.shape[-2:]  # Shape of prototype masks

        # Build masks
        gt_masks = torch.zeros((B, N, Hm, Wm), dtype=torch.float32, device=device)
        gt_classes = torch.full((B, N), -1, dtype=torch.long, device=device)  # -1 = ignore

        for idx in range(det_targets.shape[0]):
            img_idx = image_indices[idx]
            cls = class_ids[idx]
            x1, y1, x2, y2 = boxes[idx].round().int()

            # Clip to valid bounds
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, Wm - 1), min(y2, Hm - 1)
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes

            # Assign first available slot in gt_masks for this image
            slot = (gt_classes[img_idx] == -1).nonzero(as_tuple=True)[0]
            if len(slot) == 0:
                continue  # No slot available
            slot = slot[0]

            gt_masks[img_idx, slot, y1:y2, x1:x2] = 1.0
            gt_classes[img_idx, slot] = cls

        # Compute predicted masks
        pred_masks = torch.einsum("bnc,bchw->bnhw", mask_coeffs, proto_masks)  # [B, N, H, W]

        # Filter valid slots (those where gt_classes != -1)
        valid = gt_classes != -1
        if valid.sum() == 0:
            mask_loss = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)
        else:
            pred_masks_valid = pred_masks[valid]
            gt_masks_valid = gt_masks[valid]
            mask_loss = self.losses[0](pred_masks_valid, gt_masks_valid)

            class_scores_valid = class_scores[valid]
            gt_classes_valid = gt_classes[valid]
            class_loss = self.losses[1](class_scores_valid, gt_classes_valid)
        print("det_pred shape:", det_pred.shape, "nm:", nm, "C:", C)

        # ========================
        # 2. Semantic Segmentation
        # ========================
        seg_pred = predictions[1]  # [B, num_classes, H, W]
        seg_gt = targets[1]  # [B, H, W]

        seg_loss = self.losses[2](seg_pred, seg_gt)
        seg_gt_flat = seg_gt.view(-1)
        seg_loss = self.losses[2](seg_loss, seg_gt_flat)
        print("mask_loss:", mask_loss.item(), "class_loss:", class_loss.item(), "seg_loss:", seg_loss.item())

        # ========================
        # 3. Conditional logic
        # ========================
        if cfg.TRAIN.ENC_SEG_ONLY or cfg.TRAIN.SEG_ONLY:
            mask_loss = 0 * mask_loss
            class_loss = 0 * class_loss
        if cfg.TRAIN.DRIVABLE_ONLY:
            mask_loss = 0 * mask_loss
            class_loss = 0 * class_loss

        # ========================
        # 4. Final Loss
        # ========================
        loss = (
                self.lambdas[0] * mask_loss +
                self.lambdas[1] * class_loss +
                self.lambdas[2] * seg_loss
        )

        return loss, (mask_loss.item(), class_loss.item(), seg_loss.item(), loss.item())


def get_loss(cfg, device):
    """
    Returns the multi-head loss with instance + semantic segmentation.
    """
    BCE_mask = nn.BCEWithLogitsLoss().to(device)
    CE_class = nn.CrossEntropyLoss().to(device)
    BCE_seg = nn.CrossEntropyLoss(ignore_index=255)  #nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)

    loss_list = [BCE_mask, CE_class, BCE_seg]
    return MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)


# example
# class L1_Loss(nn.Module)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
