#!/usr/bin/env python3

import argparse
import os, sys
from pathlib import Path
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.models import get_net
from lib.config import cfg, update_config
from lib.utils.utils import select_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="semantics_submission")
    parser.add_argument("--modelDir", type=str, default="", help="model directory")
    parser.add_argument("--logDir", type=str, default="runs/", help="log directory")
    return parser.parse_args()


def main():
    args = parse_args()
    update_config(cfg, args)

    device = select_device(None, batch_size=2)

    # model
    model = get_net(cfg)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # PHENO expects 3 sem classes: 0,1,2
    num_classes = cfg.num_seg_class  # should be 3

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    os.makedirs(args.out_dir, exist_ok=True)

    img_paths = sorted([p for p in os.listdir(args.test_dir) if p.lower().endswith(".png")])

    print(f"> processing {len(img_paths)} images")

    for fname in img_paths:
        p = os.path.join(args.test_dir, fname)
        img0 = cv2.imread(p)
        h0,w0 = img0.shape[:2]

        img = cv2.resize(img0, tuple(cfg.MODEL.IMAGE_SIZE))
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            det_out, da_out = model(img)

        # da_out shape: [1, num_classes, H, W]
        # argmax
        da = da_out.argmax(1)[0].cpu().numpy().astype(np.uint8)

        # resize back
        da = cv2.resize(da, (w0,h0), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(args.out_dir, fname), da)

    print("\n✅ DONE")
    print(f"→ semantic PNGs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
