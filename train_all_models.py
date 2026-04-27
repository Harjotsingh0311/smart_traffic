# train_all_models.py
# Trains all models SEQUENTIALLY on one GPU (one finishes → next starts)
# RT-DETR uses AdamW + batch=8 (different from YOLO models)
import torch
from ultralytics import YOLO

# ══════════════════════════════════════════
#  CHANGE THIS to combined3 after merging
# ══════════════════════════════════════════
DATA    = 'dataset/combined3/data.yaml'
EPOCHS  = 25
IMGSZ   = 480
DEVICE  = 0
WORKERS = 2

# (model_file, run_name, batch, optimizer, lr)
# RT-DETR needs AdamW + smaller batch — everything else uses SGD
MODELS = [
   # ('yolov8s.pt',  'traffic_yolov8n',  16, 'SGD',   0.01),
   # ('yolo11s.pt',  'traffic_yolo11s',  16, 'SGD',   0.01),
    ('rtdetr-l.pt', 'traffic_rtdetr_l',  2, 'AdamW', 0.0001),
]

if __name__ == '__main__':
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Dataset        : {DATA}")
    print(f"Training {len(MODELS)} models sequentially...\n")

    results_summary = []

    for model_file, run_name, batch, optimizer, lr in MODELS:
        print(f"\n{'='*55}")
        print(f"  Training : {model_file}")
        print(f"  Run name : {run_name}")
        print(f"  Batch    : {batch}   Optimizer: {optimizer}   LR: {lr}")
        print(f"{'='*55}\n")

        model = YOLO(model_file)

        train_args = dict(
            data          = DATA,
            epochs        = EPOCHS,
            imgsz         = IMGSZ,
            batch         = batch,
            device        = DEVICE,
            workers       = WORKERS,
            patience      = 10,
            save          = True,
            project       = 'runs/train',
            name          = run_name,
            pretrained    = True,
            optimizer     = optimizer,
            lr0           = lr,
            weight_decay  = 0.0005 if optimizer == 'SGD' else 0.0001,
            momentum      = 0.937  if optimizer == 'SGD' else 0.9,
            # Augmentation (same for all)
            degrees       = 15.0,
            scale         = 0.5,
            fliplr        = 0.5,
            hsv_h         = 0.015,
            hsv_s         = 0.7,
            hsv_v         = 0.4,
            mosaic        = 1.0,
            verbose       = True,
        )

        try:
            results = model.train(**train_args)

            metrics  = results.results_dict
            map50    = metrics.get('metrics/mAP50(B)',    0)
            map5095  = metrics.get('metrics/mAP50-95(B)', 0)

            results_summary.append({
                'model':    model_file,
                'run':      run_name,
                'mAP50':    round(map50   * 100, 1),
                'mAP5095':  round(map5095 * 100, 1),
                'status':   '✅ Done'
            })
            print(f"\n✅  {model_file} finished!")
            print(f"    mAP50    : {map50*100:.1f}%")
            print(f"    mAP50-95 : {map5095*100:.1f}%")
            print(f"    Saved at : runs/train/{run_name}/weights/best.pt")

        except Exception as e:
            print(f"\n❌  {model_file} FAILED: {e}")
            results_summary.append({
                'model':   model_file,
                'run':     run_name,
                'mAP50':   0,
                'mAP5095': 0,
                'status':  f'❌ Failed: {e}'
            })
            # Continue training next model even if one fails
            continue

    # ── Final summary table ────────────────────────────
    print(f"\n\n{'='*55}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<20} {'mAP50':>8} {'mAP50-95':>10}  Status")
    print(f"{'-'*55}")

    # Show your already-trained YOLOv8s baseline for comparison
    print(f"{'yolov8s (old)':<20} {'68.9':>8} {'47.2':>10}  ✅ Pre-trained")

    for r in results_summary:
        print(
            f"{r['model']:<20} "
            f"{r['mAP50']:>7}% "
            f"{r['mAP5095']:>9}%  "
            f"{r['status']}"
        )

    print(f"\n✅  All done! Best models are in runs/train/")
    print("    Copy the best one to:  models/best.pt")
    print("\n    Example:")
    print("    copy runs\\train\\traffic_yolo11s\\weights\\best.pt models\\best.pt")