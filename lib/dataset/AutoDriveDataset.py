import os
import json
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class MultitaskWeedsDataset(Dataset):
    def __init__(self, cfg, is_train=True, inputsize=(600, 600), transform=None):
        self.cfg = cfg
        self.inputsize = inputsize
        self.transform = transform
        self.is_train = is_train

        # Determine split
        self.split = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET

        # Paths
        self.root_dir = Path(cfg.DATASET.DATAROOT)
        self.image_dir = self.root_dir / "images" / self.split
        self.semantic_dir = self.root_dir / "semantics" / self.split
        self.instance_dir = self.root_dir / "instances" / self.split

        # Load COCO-style annotations
        json_files = list(self.instance_dir.glob("*.json"))
        assert len(json_files) == 1, f"Expected 1 json file in {self.instance_dir}, found {len(json_files)}"
        self.coco = COCO(json_files[0])

        # Get image IDs
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]

        # Base name (without extension)
        base_name = Path(image_info['file_name']).stem

        # Load individual multispectral bands (order: R, G, B, NIR, RE)
        red   = np.array(Image.open(self.image_dir / f"{base_name}_R.png"))
        green = np.array(Image.open(self.image_dir / f"{base_name}_G.png"))
        blue  = np.array(Image.open(self.image_dir / f"{base_name}_B.png"))
        nir   = np.array(Image.open(self.image_dir / f"{base_name}_NIR.png"))
        re    = np.array(Image.open(self.image_dir / f"{base_name}_RE.png"))

        # Stack into (5, H, W)
        image = np.stack([red, green, blue, nir, re], axis=0).astype(np.float32) / 255.0

        # Optional normalization (per-channel)
        image = (image - image.mean(axis=(1, 2), keepdims=True)) / (image.std(axis=(1, 2), keepdims=True) + 1e-6)

        # Load semantic segmentation mask
        semantic_path = self.semantic_dir / f"{base_name}.png"
        semantic = np.array(Image.open(semantic_path).convert('L'), dtype=np.int64)

        # Load instance annotations (bounding boxes and class IDs)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        bboxes = []
        class_ids = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]  # xyxy format
            bboxes.append(bbox)
            class_ids.append(ann['category_id'])

        bboxes = np.array(bboxes, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int64)

        # Convert to tensors
        image = torch.from_numpy(image).float()             # (5, H, W)
        semantic = torch.from_numpy(semantic).long()        # (H, W)
        bboxes = torch.from_numpy(bboxes).float()           # (N, 4)
        class_ids = torch.from_numpy(class_ids).long()      # (N,)

        sample = {
            'image': image,
            'semantic': semantic,
            'bboxes': bboxes,
            'labels': class_ids,
            'image_id': image_id,
            'image_path': str(self.image_dir / f"{base_name}_*.png"),
        }

        # Apply transforms (if any)
        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        imgs, det_labels, seg_labels, paths, shapes = [], [], [], [], []

        for i, sample in enumerate(batch):
            img = sample["image"]
            semantic = sample["semantic"]
            bboxes = sample["bboxes"]
            class_ids = sample["labels"]

            # Instance segmentation: [image_idx, class_id, x1, y1, x2, y2]
            det = torch.zeros((len(bboxes), 6), dtype=torch.float32)
            if len(bboxes):
                det[:, 0] = i  # batch index
                det[:, 1] = class_ids
                det[:, 2:] = bboxes

            imgs.append(img)
            det_labels.append(det)
            seg_labels.append(semantic)
            paths.append(sample["image_path"])
            shapes.append(img.shape[1:])  # (H, W)

        # Final format:
        return torch.stack(imgs, 0), (torch.cat(det_labels, 0), torch.stack(seg_labels, 0)), paths, shapes


