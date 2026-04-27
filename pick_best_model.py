# pick_best_model.py
import os
import shutil
import csv

RUNS_DIR   = 'runs/detect/runs/train'   # ✅ your actual path
OUTPUT_DIR = 'models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ your actual trained run names
RUN_NAMES = {
    'traffic_yolov8n':  'YOLOv8n',
    'traffic_yolo11s':  'YOLO11s',
    'traffic_rtdetr_l3': 'RT-DETR-L',
}

def get_map50(run_path):
    csv_path = os.path.join(run_path, 'results.csv')
    if not os.path.exists(csv_path):
        return 0.0
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        if not rows:
            return 0.0
        col = next((c for c in rows[0].keys()
                    if 'mAP50' in c and '95' not in c), None)
        if col is None:
            return 0.0
        best = max(float(r[col].strip()) for r in rows
                   if r[col].strip())
        return best
    except Exception as e:
        print(f"  Warning reading {csv_path}: {e}")
        return 0.0


print("=" * 55)
print("  MODEL COMPARISON")
print("=" * 55)
print(f"{'Run':<24} {'Model':<14} {'mAP50':>8}")
print("-" * 55)

results = []

for run_name, model_label in RUN_NAMES.items():
    run_path = os.path.join(RUNS_DIR, run_name)
    best_pt  = os.path.join(run_path, 'weights', 'best.pt')

    if not os.path.exists(best_pt):
        print(f"  {run_name:<24} — not found, skipping")
        continue

    map50 = get_map50(run_path)
    size  = os.path.getsize(best_pt) / (1024 * 1024)

    print(f"  {run_name:<24} {model_label:<14} "
          f"{map50*100:>6.1f}%   ({size:.1f} MB)")

    results.append({
        'run':   run_name,
        'label': model_label,
        'map50': map50,
        'pt':    best_pt,
    })

if not results:
    print("\n❌ No trained models found.")
    print(f"   Looked in: {RUNS_DIR}")
    exit()

best = max(results, key=lambda r: r['map50'])

print(f"\n{'='*55}")
print(f"  🏆 WINNER: {best['label']}  "
      f"(mAP50 = {best['map50']*100:.1f}%)")
print(f"{'='*55}")

dest = os.path.join(OUTPUT_DIR, 'best.pt')
shutil.copy(best['pt'], dest)
print(f"\n✅ Copied → {dest}")
print(f"   Run next:  python main.py")

# Save comparison summary
summary_path = os.path.join(OUTPUT_DIR, 'model_comparison.txt')
with open(summary_path, 'w') as f:
    f.write("MODEL COMPARISON SUMMARY\n")
    f.write("=" * 40 + "\n")
    for r in sorted(results, key=lambda x: x['map50'], reverse=True):
        winner = " <- SELECTED" if r['run'] == best['run'] else ""
        f.write(f"{r['label']:<14} mAP50: {r['map50']*100:.1f}%{winner}\n")
    f.write("\nRT-DETR-L: Excluded — 6GB VRAM insufficient\n")

print(f"✅ Summary saved → {summary_path}")