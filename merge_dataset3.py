# merge_dataset3.py
# Merges: Indian + Traffic + Ambulance + Truck datasets
# Final 6 classes: priority_vehicle, auto-rickshaw, bus, car, motorcycle, truck
import os
import shutil
import glob
import yaml

# ── Paths ──────────────────────────────────────────────
INDIAN_DIR    = "dataset/indian"
TRAFFIC_DIR   = "dataset/traffic"
AMBULANCE_DIR = "dataset/ambulance"
TRUCK_DIR     = "dataset/truck"     # copy downloaded folder here
OUTPUT_DIR    = "dataset/combined3"

# ── Final 6 classes ────────────────────────────────────
FINAL_CLASSES = [
    'priority_vehicle',  # 0  (ambulance / police / emergency)
    'auto-rickshaw',     # 1
    'bus',               # 2
    'car',               # 3
    'motorcycle',        # 4
    'truck',             # 5
]

# ── Class mappings ─────────────────────────────────────

# Indian dataset (nc=7):
# 0=ambulance, 1=auto-rikshaw, 2=bike, 3=bus, 4=car,
# 5=police vehicle, 6=truck
INDIAN_MAP = {
    0: 0,     # ambulance       → priority_vehicle
    1: 1,     # auto-rikshaw    → auto-rickshaw
    2: 4,     # bike            → motorcycle
    3: 2,     # bus             → bus
    4: 3,     # car             → car
    5: 0,     # police vehicle  → priority_vehicle (was skipped before!)
    6: 5,     # truck           → truck
}

# Traffic dataset (nc=3):
# 0=bus, 1=car, 2=motorcycle
TRAFFIC_MAP = {
    0: 2,     # bus        → bus
    1: 3,     # car        → car
    2: 4,     # motorcycle → motorcycle
}

# Ambulance dataset (nc=2):
# 0=ambulance, 1=emergency-vehicle
AMBULANCE_MAP = {
    0: 0,     # ambulance         → priority_vehicle
    1: 0,     # emergency-vehicle → priority_vehicle
}

# Truck dataset (nc=1):
# 0=truck
TRUCK_MAP = {
    0: 5,     # truck → truck
}

# ── Create output folders ──────────────────────────────
for split in ['train', 'valid', 'test']:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

print("✅ Output folders created")


# ── Copy and remap function ────────────────────────────
def copy_remap(img_dir, lbl_dir, dst_split, class_map, prefix):
    if not os.path.exists(img_dir):
        print(f"  [{prefix}] {dst_split}: folder not found — skipping")
        return 0

    images  = glob.glob(f"{img_dir}/*.*")
    copied  = 0
    skipped = 0

    for img_path in images:
        filename  = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        lbl_path  = os.path.join(lbl_dir, name + ".txt")

        if not os.path.exists(lbl_path):
            skipped += 1
            continue

        new_lines = []
        with open(lbl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts   = line.split()
                cls_id  = int(parts[0])
                new_cls = class_map.get(cls_id, None)
                if new_cls is None:
                    continue
                new_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        if not new_lines:
            skipped += 1
            continue

        new_img = f"{prefix}_{filename}"
        new_lbl = f"{prefix}_{name}.txt"

        shutil.copy(img_path,
            f"{OUTPUT_DIR}/{dst_split}/images/{new_img}")

        with open(
            f"{OUTPUT_DIR}/{dst_split}/labels/{new_lbl}", 'w'
        ) as f:
            f.write('\n'.join(new_lines))

        copied += 1

    print(f"  [{prefix}] {dst_split}: {copied} copied, {skipped} skipped")
    return copied


# ── Process datasets ───────────────────────────────────
print("\n📂 Processing Indian Vehicle dataset...")
for split in ['train', 'valid', 'test']:
    copy_remap(
        f"{INDIAN_DIR}/{split}/images",
        f"{INDIAN_DIR}/{split}/labels",
        split, INDIAN_MAP, "ind"
    )

print("\n📂 Processing Traffic Vehicle dataset...")
for split in ['train', 'valid']:
    copy_remap(
        f"{TRAFFIC_DIR}/{split}/images",
        f"{TRAFFIC_DIR}/{split}/labels",
        split, TRAFFIC_MAP, "trf"
    )

print("\n📂 Processing Ambulance / Emergency dataset...")
for split in ['train', 'valid', 'test']:
    copy_remap(
        f"{AMBULANCE_DIR}/{split}/images",
        f"{AMBULANCE_DIR}/{split}/labels",
        split, AMBULANCE_MAP, "amb"
    )

print("\n📂 Processing Truck dataset...")
for split in ['train', 'valid', 'test']:
    copy_remap(
        f"{TRUCK_DIR}/{split}/images",
        f"{TRUCK_DIR}/{split}/labels",
        split, TRUCK_MAP, "trk"
    )

# ── Final counts ───────────────────────────────────────
print("\n📊 Final dataset counts:")
total_all = 0
for split in ['train', 'valid', 'test']:
    imgs = len(glob.glob(f"{OUTPUT_DIR}/{split}/images/*"))
    lbls = len(glob.glob(f"{OUTPUT_DIR}/{split}/labels/*"))
    print(f"  {split}: {imgs} images, {lbls} labels")
    total_all += imgs
print(f"  TOTAL: {total_all} images")

# ── Write data.yaml ────────────────────────────────────
data = {
    'train': '../train/images',
    'val':   '../valid/images',
    'test':  '../test/images',
    'nc':    len(FINAL_CLASSES),
    'names': FINAL_CLASSES
}

with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print(f"\n✅ data.yaml written")
print(f"✅ Classes: {FINAL_CLASSES}")
print(f"✅ Combined dataset ready at: {OUTPUT_DIR}")
print("\n🚦 Ready to train!")