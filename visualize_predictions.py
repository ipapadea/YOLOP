import os
import cv2
import random
from pathlib import Path

# --- Configuration ---
images_dir = "/media/beast/Storage/ilias/precision_agriculture/PhenoBench-v110_new/PhenoBench/test/images"
pred_dir = "/media/beast/Storage/ilias/YOLOP/submission/plant_bboxes"
output_dir = "/media/beast/Storage/ilias/YOLOP/submission/visualizations"

os.makedirs(output_dir, exist_ok=True)

# class mapping (update if needed)
names = {0: "crop", 1: "weed"}
colors = {0: (0, 255, 0), 1: (255, 0, 0)}  # green=crop, blue=weed


def draw_boxes(img_path, txt_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Cannot open {img_path}")
        return

    h, w = img.shape[:2]

    if not os.path.exists(txt_path):
        print(f"[WARNING] Missing predictions for {Path(img_path).stem}")
        return

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls_id, x, y, bw, bh, conf = parts
        cls_id = int(cls_id)
        x, y, bw, bh, conf = map(float, (x, y, bw, bh, conf))

        # denormalize xywh → xyxy
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        color = colors.get(cls_id, (255, 255, 255))

        # draw rectangle and text
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(out_path, img)


# --- visualize all or a random subset ---
all_imgs = sorted([f for f in os.listdir(images_dir) if f.endswith((".png", ".jpg"))])
# sampled_imgs = random.sample(all_imgs, min(50, len(all_imgs)))  # visualize up to 50 random images

for fname in all_imgs:
    img_path = os.path.join(images_dir, fname)
    txt_path = os.path.join(pred_dir, Path(fname).stem + ".txt")
    out_path = os.path.join(output_dir, fname)
    draw_boxes(img_path, txt_path, out_path)

print(f"✅ Visualization done! Saved {len(all_imgs)} images to {output_dir}")
