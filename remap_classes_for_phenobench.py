import os

# --- Configuration ---
input_dir = "/media/beast/Storage/ilias/YOLOP/submission/plant_bboxes"
backup_dir = input_dir + "_backup"

# Create backup folder before overwriting
os.makedirs(backup_dir, exist_ok=True)

# Mapping: your IDs → PhenoBench IDs
# 0=crop → 1=crop, 1=weed → 2=weed
id_map = {0: 1, 1: 2}

count_changed = 0

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".txt"):
        continue

    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(input_dir, fname)
    backup_path = os.path.join(backup_dir, fname)

    # Backup original file
    with open(in_path, "r") as f:
        lines = f.readlines()
    with open(backup_path, "w") as f:
        f.writelines(lines)

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue

        cls_id = int(parts[0])
        if cls_id in id_map:
            cls_id = id_map[cls_id]
            parts[0] = str(cls_id)
            count_changed += 1

        new_lines.append(" ".join(parts) + "\n")

    with open(out_path, "w") as f:
        f.writelines(new_lines)

print(f"✅ Done! Updated {count_changed} detections in '{input_dir}'.")
print(f"🗂️ Backup of original files saved in: {backup_dir}")
