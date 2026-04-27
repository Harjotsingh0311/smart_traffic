# confusion_matrix.py
from ultralytics import YOLO
import os

MODEL_PATH = 'models/best.pt'
DATA_YAML  = 'dataset/combined3/data.yaml'

EXPECTED_CLASSES = [
    'priority_vehicle',
    'auto-rickshaw',
    'bus', 'car', 'motorcycle', 'truck',
]

if __name__ == '__main__':   # ✅ Windows multiprocessing fix

    print("=" * 55)
    print("  STEP 1 — Verify model class names")
    print("=" * 55)

    model = YOLO(MODEL_PATH)
    model_classes = list(model.names.values())

    print(f"\nModel classes:")
    for i, name in enumerate(model_classes):
        match = "✅" if i < len(EXPECTED_CLASSES) and name == EXPECTED_CLASSES[i] else "❌ MISMATCH"
        exp   = EXPECTED_CLASSES[i] if i < len(EXPECTED_CLASSES) else "???"
        print(f"  [{i}] {name:<22} Expected: {exp:<22} {match}")

    print("\n" + "=" * 55)
    print("  STEP 2 — Validation + Confusion Matrix")
    print("=" * 55)

    results = model.val(
        data    = DATA_YAML,
        imgsz   = 640,
        conf    = 0.40,
        iou     = 0.50,
        device  = 0,
        workers = 0,         # ✅ fixes Windows multiprocessing crash
        plots   = True,
        project = 'runs/val',
        name    = 'confusion',
        verbose = True,
    )

    print("\n" + "=" * 55)
    print("  STEP 3 — Results")
    print("=" * 55)

    map50   = results.box.map50
    map5095 = results.box.map
    mp      = results.box.mp
    mr      = results.box.mr

    print(f"\n  mAP50      : {map50*100:.1f}%")
    print(f"  mAP50-95   : {map5095*100:.1f}%")
    print(f"  Precision  : {mp*100:.1f}%")
    print(f"  Recall     : {mr*100:.1f}%")

    print(f"\n  Per-class mAP50:")
    print(f"  {'Class':<22} {'mAP50':>8}")
    print(f"  {'-'*32}")
    if hasattr(results.box, 'maps') and results.box.maps is not None:
        for cls_map, cls_name in zip(results.box.maps, model_classes):
            bar = '█' * int(cls_map * 20)
            print(f"  {cls_name:<22} {cls_map*100:>6.1f}%  {bar}")

    print(f"\n✅ Confusion matrix → runs/val/confusion/confusion_matrix.png")
    print(f"✅ Normalized matrix → runs/val/confusion/confusion_matrix_normalized.png")