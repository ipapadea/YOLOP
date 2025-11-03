#!/usr/bin/env python3
"""
Convert PhenoBench annotations (semantics + plant_instances)
into BDD100K-style JSON detection files (crop + weed).
"""

import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path


def one_anno_bdd(sem_path: Path, id_path: Path, out_dir: Path):
    """Convert one pair of semantic + instance mask to a BDD100K JSON."""
    out_file_path = out_dir / f"{sem_path.stem}.json"

    # Always convert Path to str for cv2
    id_mask = cv2.imread(str(id_path), cv2.IMREAD_UNCHANGED)
    if id_mask is None:
        print(f"[WARN] Could not read {id_path}")
        return 0
    h, w = id_mask.shape[:2]

    sem_mask = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
    if sem_mask is None:
        print(f"[WARN] Could not read {sem_path}")
        return 0

    # Merge partial classes (as in official converter)
    sem_mask[sem_mask == 3] = 1  # partial crop → crop
    sem_mask[sem_mask == 4] = 2  # partial weed → weed

    plant_ids = np.unique(id_mask)
    soil_mask = sem_mask == 0
    id_mask[soil_mask] = 0

    objects = []
    obj_id = 0

    for plant_id in plant_ids:
        if plant_id == 0:
            continue  # skip background

        if not np.any(id_mask == plant_id):
            continue

        plant_class_mask = sem_mask[id_mask == plant_id]
        for plant_class in np.unique(plant_class_mask):
            if plant_class == 0:
                continue  # skip soil

            plant_mask = np.logical_and(id_mask == plant_id, sem_mask == plant_class)
            ys, xs = np.where(plant_mask)
            if len(xs) < 3 or len(ys) < 3:
                continue

            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            obj = {
                "category": "crop" if plant_class == 1 else "weed",
                "id": obj_id,
                "attributes": {
                    "occluded": False,
                    "truncated": False,
                    "trafficLightColor": "none"
                },
                "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            }
            objects.append(obj)
            obj_id += 1

    entry = {
        "name": sem_path.stem,
        "frames": [{"timestamp": 10000, "objects": objects}],
        "attributes": {
            "weather": "clear",
            "scene": "agricultural field",
            "timeofday": "daytime"
        }
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file_path, "w") as f:
        json.dump(entry, f, indent=4)

    return len(objects)


def convert_one_split(in_dir: str, out_dir: str, split_tag: str):
    """Convert one split (train/val) to BDD100K JSONs."""
    split_dir = Path(in_dir) / split_tag
    sem_dir = split_dir / "semantics"
    id_dir = split_dir / "plant_instances"
    output_dir = Path(out_dir) / split_tag / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    sem_files = sorted([f for f in sem_dir.glob("*.png")])
    print(f"Converting {split_tag}: {len(sem_files)} semantic masks found")

    total_objects = 0
    for sem_path in tqdm(sem_files):
        id_path = id_dir / sem_path.name
        total_objects += one_anno_bdd(sem_path, id_path, output_dir)

    print(f"✅ Finished {split_tag}: {len(sem_files)} images, {total_objects} total boxes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PhenoBench to BDD100K-style JSONs")
    parser.add_argument("--input_dir", required=True, help="Path to PhenoBench root")
    parser.add_argument("--output_dir", required=True, help="Path to save converted JSONs")
    args = parser.parse_args()

    for split_tag in ["train", "val"]:
        convert_one_split(args.input_dir, args.output_dir, split_tag)
