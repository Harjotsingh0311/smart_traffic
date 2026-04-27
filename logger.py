# logger.py
import csv
import os
from datetime import datetime


class TrafficLogger:
    def __init__(self, log_file='logs/detections.csv'):
        os.makedirs('logs', exist_ok=True)
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'lane',
                    'car', 'bus', 'truck',
                    'motorcycle', 'auto_rickshaw',
                    'priority_vehicle',
                    'total_vehicles',
                    'congestion',
                    'green_lane',
                    'green_duration_sec',
                    'inference_latency_ms',
                    'fps'
                ])

    def log(self, lane, counts, congestion,
            green_lane, green_duration,
            latency_ms=0, fps=0):
        total = sum(counts.values())
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                lane,
                counts.get('car',              0),
                counts.get('bus',              0),
                counts.get('truck',            0),
                counts.get('motorcycle',       0),
                counts.get('auto-rickshaw',    0),
                counts.get('priority_vehicle', 0),
                total,
                congestion,
                green_lane,
                green_duration,
                round(latency_ms, 2),
                round(fps, 2)
            ])