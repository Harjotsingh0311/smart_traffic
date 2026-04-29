# 🚦 IoT-Driven Smart Traffic Management System
### Edge AI Based | TIET Patiala | Harjot Singh (1024240039)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.36-orange)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+cu118-red)](https://pytorch.org)

Real-time traffic intersection management using YOLO11s custom-trained on Indian traffic data. Detects 6 vehicle classes across 4 lanes simultaneously, adaptively controls signal timing, and grants immediate green wave to priority vehicles.

---

## 🎯 Features

- **4-Lane Simultaneous Detection** — dedicated video feed per lane (RIGHT / DOWN / LEFT / UP)
- **6 Vehicle Classes** — priority_vehicle, auto-rickshaw, bus, car, motorcycle, truck
- **Adaptive Signal Timing** — green duration calculated from live vehicle density
- **Priority Vehicle Green Wave** — 95% recall, immediate signal override
- **Auto-Rotating Signals** — lanes cycle automatically based on congestion
- **Live Web Dashboard** — Flask at `localhost:5000`, auto-refresh every 2s
- **Pygame Simulation** — 2D intersection view synchronized with detections
- **CSV Logging** — per-frame detection log with timestamps and metrics
- **Edge Deployment Ready** — ONNX export for Jetson Nano 

---

## 📊 Model Performance

| Model | mAP@0.5 | Size | Status |
|---|---|---|---|
| YOLOv8n | 83.6% | 21.5 MB | Baseline |
| **YOLO11s** | **83.4%** | **18.3 MB** | ✅ **Deployed** |
| RT-DETR-L | 55.6% | 113 MB | Excluded (6GB VRAM insufficient) |

### Per-Class AP — YOLO11s (Deployed)

| Class | AP@0.5 | Notes |
|---|---|---|
| truck | 99.3% | Best performing class |
| priority_vehicle | 96.4% | Critical — excellent for green wave |
| bus | 84.7% | Good |
| auto-rickshaw | 80.4% | Limited by training data (117 instances) |
| motorcycle | 71.1% | Small object challenge |
| car | 68.4% | Background confusion — dataset imbalance |

---

## 🗂️ Project Structure

```
smart_traffic/
├── dataset/combined3/       ← Merged training dataset (4 sources)
├── models/best.pt/          ← Deployed YOLO11s model
├── logs/detections.csv      ← Per-frame detection log
├── images/                  ← Pygame simulation assets
├── main.py                  ← Core detection & signal control
├── signal_time.py           ← Adaptive signal timing controller
├── logger.py                ← CSV logger
├── simulation.py            ← Pygame intersection simulation
├── run_all.py               ← Launch all components
├── train_all_models.py      ← Sequential model training
├── merge_dataset3.py        ← Dataset merge & class remapping
├── confusion_matrix.py      ← Post-training validation
├── pick_best_model.py       ← Auto model selection by mAP50
├── export_model.py          ← ONNX export for Jetson Nano
├── analytics.py             ← Log charts
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# Create environment
conda create -n traffic_env python=3.10
conda activate traffic_env

# Navigate to project
cd "C:\workstation\3RD & 4th SEM\YOLO\PROJECT\smart_traffic"

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
# Run everything at once
python run_all.py

# Or run separately:
python main.py        # Detection window + dashboard
python simulation.py  # Pygame intersection simulation
```

Open `http://localhost:5000` for the live dashboard.

**Detection window controls:** `Q` = Quit, `R` = Restart videos

---

## 🏋️ Training Pipeline

```bash
# 1. Merge 4 datasets into combined3
python merge_dataset3.py

# 2. Train all models sequentially
python train_all_models.py

# 3. Auto-select best model
python pick_best_model.py
# → copies winner to models/best.pt

# 4. Generate confusion matrix
python confusion_matrix.py

# 5. Export for Jetson Nano
python export_model.py
# → models/best.onnx
```

---

## 🗃️ Dataset — combined3

4 datasets merged with class remapping:

| Source Dataset | Original Class | Mapped To |
|---|---|---|
| Indian Vehicle | ambulance, police vehicle | priority_vehicle (0) |
| Indian Vehicle | auto-rikshaw | auto-rickshaw (1) |
| Indian Vehicle | bike | motorcycle (4) |
| Ambulance dataset | ambulance, emergency-vehicle | priority_vehicle (0) |
| Traffic dataset | bus, car, motorcycle | bus(2), car(3), motorcycle(4) |
| Truck dataset | truck | truck (5) |

**Final 6 Classes:**
```
0: priority_vehicle   (ambulance / police / school van)
1: auto-rickshaw
2: bus
3: car
4: motorcycle
5: truck
```

**Instance counts:**
```
car:              39,607  (66.3%)
motorcycle:        7,978  (13.4%)
bus:               5,438   (9.1%)
priority_vehicle:  3,469   (5.8%)
truck:             2,463   (4.1%)
auto-rickshaw:       117   (0.2%)
```

---

## 🧠 Signal Control Logic

```
Priority Order:
1. Priority vehicle detected  →  immediate green wave to that lane
2. Auto-rotate timer active   →  currently scheduled lane
3. Fallback                   →  highest vehicle count lane

Green Duration:
T = ceil((cars×2 + buses×3 + trucks×3 + bikes×1 + rickshaws×1.5)
         / (noOfLanes + 1))
T = clamp(T, 10s, 60s)
```

---

## 🔧 Key Configuration (main.py)

```python
CONF_THRESH   = 0.30   # general detection confidence
PV_THRESH     = 0.65   # priority vehicle threshold (stricter)
NMS_IOU       = 0.45   # non-maximum suppression threshold
PROCESS_EVERY = 2      # inference every N frames

LANE_VIDEOS = {
    'RIGHT': 'lane_right.mp4',
    'DOWN':  'lane_down.mp4',
    'LEFT':  'lane_left.mp4',
    'UP':    'lane_up.mp4',
}
```

---

## 📡 Edge Deployment — Jetson Nano

```bash
# Export to ONNX
python export_model.py
# Output: models/best.onnx

# On Jetson Nano (via RealVNC):
# Deploy best.onnx or best.pt
```

---

## 📈 Analytics

```bash
python analytics.py
# Output: logs/analytics.png
# Charts: vehicle counts over time, type distribution,
#         green signal frequency per lane, inference latency
```

---

## 📋 Log Format (logs/detections.csv)

```
timestamp, lane, car, bus, truck, motorcycle, auto_rickshaw,
priority_vehicle, total_vehicles, congestion, green_lane,
green_duration_sec, inference_latency_ms, fps
```

---

## 🖥️ Hardware

| Component | Spec |
|---|---|
| Training GPU | NVIDIA RTX 3050 6GB Laptop |
| Edge Device | NVIDIA Jetson Nano 4GB |
| Remote Access | RealVNC |
| Training OS | Windows 11 + Anaconda |
| Edge OS | Ubuntu 20.04 |
| CUDA | 11.8 |

---

## 🏛️ Academic Info

**Institution:** Thapar Institute of Engineering & Technology, Patiala  
**Student:** Harjot Singh | Roll No: 1024240039 | Batch: 2X15  
**Domain:** AI · Computer Vision · Deep Learning · Edge AI · IoT · Smart Cities

---

## 📚 References

1. Ultralytics YOLO — https://github.com/ultralytics/ultralytics  
2. DETRs Beat YOLOs on Real-time Object Detection — arXiv:2304.08069  
3. NVIDIA Triton Inference Server — https://docs.nvidia.com/deeplearning/triton-inference-server/  
4. Roboflow Datasets — https://roboflow.com  
5. OpenCV — https://docs.opencv.org  
