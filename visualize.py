import json
import cv2
from pathlib import Path

# === CONFIGURATION ===
root = Path("/media/beast/Storage/ilias/precision_agriculture/PhenoBench-v110/PhenoBench")
split = "train"  # "train" or "val"
json_dir = root / "bddstyle" / split / "labels"
img_dir = root / split / "images"   # original RGB images
output_dir = root / "bddstyle" / split / "viz"
output_dir.mkdir(parents=True, exist_ok=True)

# Optional: color map per category
COLORS = {
    "crop": (0, 255, 0),    # green
    "weed": (0, 0, 255),    # red
}

def visualize_one(json_path: Path):
    # load corresponding image
    img_path = img_dir / (json_path.stem + ".png")
    if not img_path.exists():
        print(f"[WARN] Missing image for {json_path.name}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Failed to read {img_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # draw each bounding box
    for obj in data["frames"][0]["objects"]:
        cat = obj["category"]
        color = COLORS.get(cat, (255, 255, 0))
        box = obj["box2d"]
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, cat, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2, cv2.LINE_AA
        )

    # save visualization
    out_path = output_dir / f"{json_path.stem}_viz.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved visualization: {out_path}")


# === MAIN LOOP ===
# pick a few random JSONs
json_files = sorted(list(json_dir.glob("*.json")))

# visualize only first N examples
N = 10
for jpath in json_files[:N]:
    visualize_one(jpath)
