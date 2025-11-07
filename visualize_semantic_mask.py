import cv2
import numpy as np
import os

# ---------- CONFIG ----------
mask_dir = "semantics_submission"           # folder με τα mask .png
out_dir  = "semantics_submission_vis"       # που να σωθούν visualize images

os.makedirs(out_dir, exist_ok=True)

# color map phenobench
color_map = {
    0: (0, 0, 0),        # soil    -> black
    1: (0, 255, 0),      # crop    -> green
    2: (255, 0, 0),      # weed    -> red
}

files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

for f in files:
    m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_UNCHANGED)

    # make RGB canvas
    vis = np.zeros((m.shape[0], m.shape[1],3), np.uint8)
    for k, c in color_map.items():
        vis[m == k] = c

    cv2.imwrite(os.path.join(out_dir, f), vis)

print("✅ done — visualize images saved in:", out_dir)
