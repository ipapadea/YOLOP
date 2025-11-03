#!/usr/bin/env python3
"""
Generate CodaLab submission for PhenoBench test set
---------------------------------------------------
This script:
1. Loads a trained YOLOP-style multitask model
2. Runs inference on PhenoBench test images (no labels)
3. Writes YOLO-format txt files for each image under 'plant_bboxes/'
4. Zips the results into 'submission.zip'
"""

import argparse
import os
import sys
import torch
import cv2
import zipfile
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- Import YOLOP libs ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from lib.models import get_net
from lib.config import cfg, update_config
from lib.utils.utils import create_logger, select_device
from lib.core.general import non_max_suppression, scale_coords, xyxy2xywh


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PhenoBench submission zip")
    parser.add_argument("--cfg", type=str, required=True, help="path to config file")
    parser.add_argument("--weights", type=str, required=True, help="path to trained .pth checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="path to PhenoBench/test/images folder")
    parser.add_argument("--output_dir", type=str, default="submission", help="output directory for results")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--modelDir", type=str, default="", help="model directory")
    parser.add_argument("--logDir", type=str, default="runs/", help="log directory")
    return parser.parse_args()


def run_inference(model, img_path, device, conf_thres, iou_thres):
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"[WARNING] Could not read {img_path}")
        return []

    h0, w0 = img0.shape[:2]
    img = cv2.resize(img0, (512, 512))
    img = img[..., ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        det_out, _ = model(img_tensor)
        inf_out, _ = det_out
        preds = non_max_suppression(inf_out, conf_thres, iou_thres)[0]

    if preds is None or len(preds) == 0:
        return []

    preds[:, :4] = scale_coords(img_tensor.shape[2:], preds[:, :4], img0.shape).round()
    results = []
    for *xyxy, conf, cls in preds.tolist():
        x_c, y_c, w, h = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
        # normalize
        x_c /= w0
        y_c /= h0
        w /= w0
        h /= h0
        results.append((int(cls), x_c, y_c, w, h, conf))
    return results


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, "inference")
    device = select_device(logger, batch_size=4)

    # --- Build and load model ---
    model = get_net(cfg)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    # Set your detection class mapping
    model.nc = 2
    model.names = {0: "crop", 1: "weed"}

    # --- Prepare output dirs ---
    os.makedirs(args.output_dir, exist_ok=True)
    bbox_dir = os.path.join(args.output_dir, "plant_bboxes")
    os.makedirs(bbox_dir, exist_ok=True)

    img_paths = sorted([
        os.path.join(args.test_dir, f)
        for f in os.listdir(args.test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"Running inference on {len(img_paths)} test images...")
    for img_path in tqdm(img_paths):
        results = run_inference(model, img_path, device, args.conf_thres, args.iou_thres)
        txt_name = Path(img_path).stem + ".txt"
        txt_path = os.path.join(bbox_dir, txt_name)
        with open(txt_path, "w") as f:
            if len(results) == 0:
                # PhenoBench expects class 0 (soil) to exist even if no detections
                f.write("0 0.5 0.5 1.0 1.0 0.0\n")
                continue

            for cls, x, y, w, h, conf in results:
                # --- Re-map 0→1 (crop), 1→2 (weed) ---
                if cls == 0:
                    cls = 1
                elif cls == 1:
                    cls = 2
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

    # Optional description.txt
    # desc_path = os.path.join(args.output_dir, "description.txt")
    # with open(desc_path, "w") as f:
    #     f.write("name: TwinLiteNet2Scaled (YOLOP multitask)\n")
    #     f.write("pdf url: https://arxiv.org/pdf/2210.07879.pdf\n")
    #     f.write("code url: https://github.com/yourrepo\n")

    # --- Zip everything ---
    zip_path = os.path.join(args.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(args.output_dir):
            for file in files:
                if file.endswith(".zip"):
                    continue
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), args.output_dir))

    print(f"\n✅ Submission ready: {zip_path}")
    print(f"Upload this file to CodaLab.")


if __name__ == "__main__":
    main()
