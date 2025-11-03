import os
import cv2
import numpy as np
from pathlib import Path

# === CONFIGURE THESE PATHS ===
# Root folder where PhenoBench semantics are stored
# e.g., phenobench/semantics/train, phenobench/semantics/val, phenobench/semantics/test
input_root = Path("/media/beast/Storage/ilias/precision_agriculture/PhenoBench_bdd100k_style/plants_seg_annotations/val/semantics")

# Output folder for 3-channel masks (keeps same subfolder structure)
output_root = Path("/media/beast/Storage/ilias/precision_agriculture/PhenoBench_bdd100k_style/plants_seg_annotations/val/semantics_rgb")
output_root.mkdir(parents=True, exist_ok=True)

# Optional: define subfolders if they exist
subsets = [""]
for subset in subsets:
    in_dir = input_root / subset
    out_dir = output_root / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {subset} ...")

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith(".png"):
            continue

        in_path = in_dir / fname
        out_path = out_dir / fname

        # --- Read single-channel label map ---
        mask = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[WARN] Could not read {in_path}")
            continue

        # --- Create binary channels ---
        soil = (mask == 0).astype(np.uint8) * 255
        crop = (mask == 1).astype(np.uint8) * 255
        weed = (mask == 2).astype(np.uint8) * 255

        # --- Stack into 3-channel RGB-like mask ---
        stacked = np.stack([soil, crop, weed], axis=-1)

        # --- Save ---
        cv2.imwrite(str(out_path), stacked)

    print(f"✅ Finished {subset}, saved to {out_dir}")
