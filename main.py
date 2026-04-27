# main.py
import cv2
import time
import threading
import numpy as np
from collections import Counter
from ultralytics import YOLO
from signal_time import TrafficSignalController
from logger import TrafficLogger

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
MODEL_PATH    = 'models/best.pt'
CONF_THRESH   = 0.30   # lower = more detections
PV_THRESH     = 0.65   # priority vehicle confidence
NMS_IOU       = 0.45   # NMS threshold
PROCESS_EVERY = 2      # run detection every N frames
DISPLAY_W     = 960
DISPLAY_H     = 720

# 4 lane videos
LANE_VIDEOS = {
    'RIGHT': 'lane_right.mp4',
    'DOWN':  'lane_down.mp4',
    'LEFT':  'lane_left.mp4',
    'UP':    'lane_up.mp4',
}

# Class names — must match training order
CLASS_NAMES = [
    'priority_vehicle',   # 0
    'auto-rickshaw',      # 1
    'bus',                # 2
    'car',                # 3
    'motorcycle',         # 4
    'truck',              # 5
]

TRAFFIC_CLASSES = {
    'auto-rickshaw', 'bus', 'car',
    'motorcycle', 'truck'
}

LANES        = ['RIGHT', 'DOWN', 'LEFT', 'UP']
CLEAR_MAX    = 5
MODERATE_MAX = 12

# Colors per class (BGR)
CLASS_COLORS = {
    'priority_vehicle': (0,   0,   255),  # red
    'auto-rickshaw':    (0,   165, 255),  # orange
    'bus':              (255, 0,   0  ),  # blue
    'car':              (0,   255, 0  ),  # green
    'motorcycle':       (255, 0,   255),  # magenta
    'truck':            (0,   255, 255),  # yellow
}
# ══════════════════════════════════════════

# Global state
_green_idx    = 0
_last_switch  = time.time()
_current_dur  = 20

all_lane_counts = {l: Counter() for l in LANES}
lane_congestion = {l: 'CLEAR'   for l in LANES}


def get_congestion(counts):
    total = sum(v for k, v in counts.items()
                if k in TRAFFIC_CLASSES)
    if total > MODERATE_MAX:
        return 'CONGESTED', (0, 0, 255)
    elif total > CLEAR_MAX:
        return 'MODERATE',  (0, 165, 255)
    else:
        return 'CLEAR',     (0, 200, 0)


def auto_rotate(green_dur):
    """Cycle green lane every green_dur seconds."""
    global _green_idx, _last_switch, _current_dur
    now     = time.time()
    elapsed = now - _last_switch
    if elapsed >= _current_dur:
        _green_idx   = (_green_idx + 1) % 4
        _last_switch = now
        _current_dur = green_dur
        print(f"\n🟢 Signal → {LANES[_green_idx]} "
              f"for {_current_dur}s")
    return max(0, int(_current_dur - (time.time() - _last_switch)))


def draw_detections(frame, boxes_data):
    """
    Draw custom bounding boxes with class colors.
    boxes_data: list of (x1,y1,x2,y2,conf,label)
    """
    for (x1, y1, x2, y2, conf, label) in boxes_data:
        color = CLASS_COLORS.get(label, (0, 255, 0))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        txt   = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lbl_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame,
                      (x1, lbl_y - th - 4),
                      (x1 + tw + 4, lbl_y),
                      color, -1)
        cv2.putText(frame, txt,
                    (x1 + 2, lbl_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_overlay(frame, counts, active_lane,
                 green_lane, green_dur, time_remaining,
                 latency_ms, fps, priority_detected):
    h, w      = frame.shape[:2]
    total     = sum(v for k, v in counts.items()
                    if k in TRAFFIC_CLASSES)
    cong, cc  = get_congestion(counts)
    is_green  = (active_lane == green_lane)
    sig_color = (0, 220, 0) if is_green else (0, 0, 255)

    # ── Top info bar ──────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 260), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 260), (25, 25, 25), 2)

    # Row 1 — Camera + Signal
    sig_txt = 'GREEN' if is_green else 'RED'
    cv2.putText(frame,
        f"Camera: {active_lane}   "
        f"Signal: {sig_txt}   "
        f"Green: {green_lane} ({time_remaining}s left)",
        (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
        0.85, sig_color, 2, cv2.LINE_AA)

    # Row 2 — Vehicle counts
    parts = []
    pv = counts.get('priority_vehicle', 0)
    if pv > 0:
        parts.append(f"PRIORITY:{pv}")
    for k in ['car','bus','truck','motorcycle','auto-rickshaw']:
        v = counts.get(k, 0)
        if v > 0:
            parts.append(f"{k.upper()}:{v}")
    veh_str = '  '.join(parts) if parts else 'No vehicles'
    row2_col = (0, 100, 255) if priority_detected \
               else (255, 215, 0)
    cv2.putText(frame,
        f"Total: {total}   {veh_str}",
        (12, 78), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, row2_col, 2, cv2.LINE_AA)

    # Row 3 — Congestion
    cv2.putText(frame,
        f"Traffic Status: {cong}   "
        f"Green Duration: {green_dur}s",
        (12, 118), cv2.FONT_HERSHEY_SIMPLEX,
        0.85, cc, 2, cv2.LINE_AA)

    # Row 4 — All lanes summary
    lane_parts = []
    for l in LANES:
        lcount = sum(
            v for k, v in all_lane_counts[l].items()
            if k in TRAFFIC_CLASSES)
        lcong  = lane_congestion.get(l, 'CLR')[:3]
        marker = '●' if l == green_lane else '○'
        lane_parts.append(f"{marker}{l}:{lcong}({lcount})")
    cv2.putText(frame,
        '  '.join(lane_parts),
        (12, 158), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, (170, 170, 170), 1, cv2.LINE_AA)

    # Row 5 — Metrics
    cv2.putText(frame,
        f"Latency:{latency_ms:.0f}ms  "
        f"FPS:{fps:.1f}  "
        f"Model:YOLOv8s Custom  "
        f"Conf:{CONF_THRESH}",
        (12, 196), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (140, 140, 140), 1, cv2.LINE_AA)

    # Row 6 — Controls
    cv2.putText(frame,
        "Q:Quit   R:Restart   "
        "Auto-switching every green cycle",
        (12, 232), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (90, 90, 90), 1, cv2.LINE_AA)

    # ── Congestion badge — top right ──────────────
    bx = w - 240
    cv2.rectangle(frame, (bx, 8), (w-8, 68), cc, -1)
    cv2.rectangle(frame, (bx, 8), (w-8, 68),
                  (255, 255, 255), 2)
    cv2.putText(frame, cong, (bx + 10, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 0, 0), 2, cv2.LINE_AA)

    # ── Green timer bar ───────────────────────────
    bar_w  = w - 24
    bar_y  = 252
    fill   = int(bar_w * max(0, time_remaining)
                 / max(_current_dur, 1))
    cv2.rectangle(frame, (12, bar_y),
                  (12 + bar_w, bar_y + 6),
                  (35, 35, 35), -1)
    cv2.rectangle(frame, (12, bar_y),
                  (12 + fill, bar_y + 6),
                  (0, 220, 0) if is_green
                  else (0, 0, 180), -1)

    # ── Priority vehicle alert — bottom ───────────
    if priority_detected:
        cv2.rectangle(frame,
                      (0, h - 75), (w, h),
                      (0, 0, 150), -1)
        cv2.rectangle(frame,
                      (0, h - 75), (w, h),
                      (0, 0, 255), 3)
        cv2.putText(frame,
            "*** PRIORITY VEHICLE — GREEN WAVE ACTIVE ***",
            (12, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def run_detection(model, frame, lane):
    """Run YOLO on one frame, return counts + box data."""
    results = model(
        frame,
        conf=CONF_THRESH,
        iou=NMS_IOU,
        imgsz=640,
        verbose=False
    )[0]

    counts    = Counter()
    boxes_data = []
    pv_found  = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id >= len(CLASS_NAMES):
            continue
        label = CLASS_NAMES[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes_data.append((x1, y1, x2, y2, conf, label))

        if label == 'priority_vehicle':
            if conf >= PV_THRESH:
                pv_found = True
                counts[label] += 1
        else:
            counts[label] += 1

    return counts, boxes_data, pv_found


def main():
    global all_lane_counts, lane_congestion

    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print(f"✅ Classes: {list(model.names.values())}")

    # Open all 4 video captures
    caps = {}
    for lane, video in LANE_VIDEOS.items():
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"❌ Cannot open {video}")
            return
        caps[lane] = cap
        print(f"✅ {lane}: {video}")

    orig_fps = caps['RIGHT'].get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30
    print(f"\n✅ FPS: {orig_fps:.0f}")
    print("🚦 Running... Q=Quit  R=Restart\n")

    controller = TrafficSignalController()
    controller.forced_lane = 'RIGHT'
    logger     = TrafficLogger()

    # Per-lane annotated frames cache
    lane_frames = {l: None for l in LANES}

    frame_idx    = 0
    fps          = 0.0
    t_fps        = time.time()
    last_latency = 0.0
    last_priority = False
    last_green    = 'RIGHT'
    last_duration = 20
    last_remaining = 20

    cv2.namedWindow('Smart Traffic Management System',
                    cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Smart Traffic Management System',
                     DISPLAY_W, DISPLAY_H)

    delay = max(1, int(1000 / orig_fps))

    while True:
        # Read one frame from ALL 4 lanes
        frames = {}
        for lane, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frames[lane] = frame if ret else \
                np.zeros((480, 640, 3), dtype=np.uint8)

        frame_idx += 1

        # FPS counter
        if frame_idx % 30 == 0:
            fps   = 30.0 / max(time.time()-t_fps, 0.001)
            t_fps = time.time()

        # ── Detection every N frames ──────────────────
        if frame_idx % PROCESS_EVERY == 0:
            priority_detected = False
            t0 = time.time()

            for lane in LANES:
                frame = frames[lane]
                counts, boxes_data, pv = run_detection(
                    model, frame, lane)

                all_lane_counts[lane] = counts
                cong, _ = get_congestion(counts)
                lane_congestion[lane] = cong

                if pv:
                    priority_detected = True
                    controller.set_priority_vehicle(lane)

                # Draw custom boxes on this lane's frame
                annotated = frames[lane].copy()
                annotated = draw_detections(annotated, boxes_data)
                lane_frames[lane] = annotated

            last_latency = (time.time() - t0) * 1000

            if not priority_detected:
                controller.reset_priority_vehicle()

            # Signal logic
            green_lane = controller.get_green_lane(
                {l: dict(c)
                 for l, c in all_lane_counts.items()})
            green_dur  = controller.calculate_green_time(
                dict(all_lane_counts[LANES[_green_idx]]))

            # Auto-rotate
            controller.forced_lane = LANES[_green_idx]
            time_remaining = auto_rotate(green_dur)

            last_green     = green_lane
            last_duration  = green_dur
            last_priority  = priority_detected
            last_remaining = time_remaining

            # Log active lane
            active_lane   = LANES[_green_idx]
            active_counts = dict(all_lane_counts[active_lane])
            cong_active, _ = get_congestion(active_counts)

            logger.log(
                active_lane, active_counts,
                cong_active, green_lane, green_dur,
                last_latency, fps)

            # Console
            lane_summary = '  '.join(
                f"{l}:{lane_congestion[l][:3]}"
                f"({sum(v for k,v in all_lane_counts[l].items() if k in TRAFFIC_CLASSES)})"
                for l in LANES
            )
            pv_tag = ' 🚨PV' if priority_detected else ''
            print(f"[{active_lane}] "
                  f"Green→{green_lane}({time_remaining}s) | "
                  f"{lane_summary} | "
                  f"{last_latency:.0f}ms | "
                  f"{fps:.1f}fps{pv_tag}")

        else:
            # Recalculate remaining time every frame
            time_remaining = max(
                0, int(_current_dur -
                       (time.time() - _last_switch)))
            last_remaining = time_remaining

        # ── Display active (green) lane ───────────────
        active_lane = LANES[_green_idx]
        display     = lane_frames.get(active_lane)

        if display is None:
            display = frames.get(
                active_lane,
                np.zeros((480, 640, 3), dtype=np.uint8))
        else:
            display = display.copy()

        active_counts = dict(all_lane_counts[active_lane])

        display = draw_overlay(
            display,
            active_counts,
            active_lane,
            last_green,
            last_duration,
            last_remaining,
            last_latency,
            fps,
            last_priority
        )

        cv2.imshow(
            'Smart Traffic Management System',
            display)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("\n👋 Quit")
            break
        elif key == ord('r'):
            for cap in caps.values():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            print("🔄 All videos restarted")

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done. Logs → logs/detections.csv")


if __name__ == '__main__':
    main()