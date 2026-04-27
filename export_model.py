# export_model.py
from ultralytics import YOLO

if __name__ == '__main__':
    print("Loading model...")
    model = YOLO('models/best.pt')
    
    print("Classes:", model.names)
    
    print("\nExporting to ONNX...")
    model.export(
        format='onnx',
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        nms=True
    )
    
    print("\n✅ Done! File: models/best.onnx")
    print("This file goes to Jetson Nano into Triton Server")