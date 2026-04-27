# analytics.py
import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = 'logs/detections.csv'

if not os.path.exists(LOG_FILE):
    print("❌ No log file found. Run main.py first!")
    exit()

df = pd.read_csv(LOG_FILE)
print(f"✅ Loaded {len(df)} log entries")
print(df.head())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Smart Traffic Management — Analytics',
             fontsize=16, fontweight='bold')

# ── Plot 1 — Total vehicles over time ────────────────
vehicle_cols = ['priority_vehicle', 'auto_rickshaw',
                'car', 'bus', 'truck', 'motorcycle']
# Only sum columns that exist (safe for partial logs)
existing = [c for c in vehicle_cols if c in df.columns]
df['total'] = df[existing].sum(axis=1)

axes[0, 0].plot(df['total'].values, color='#38bdf8', linewidth=2)
axes[0, 0].fill_between(range(len(df)), df['total'].values,
                         alpha=0.3, color='#38bdf8')
axes[0, 0].set_title('Total Vehicles Over Time')
axes[0, 0].set_ylabel('Vehicle Count')
axes[0, 0].set_xlabel('Frame')
axes[0, 0].grid(True, alpha=0.3)

# ── Plot 2 — Vehicle type distribution (pie) ─────────
sums = {}
labels_map = {
    'car':              'Car',
    'bus':              'Bus',
    'truck':            'Truck',
    'motorcycle':       'Motorcycle',
    'auto_rickshaw':    'Auto-Rickshaw',
    'priority_vehicle': 'Priority Vehicle',
}
for col, label in labels_map.items():
    if col in df.columns:
        total = df[col].sum()
        if total > 0:
            sums[label] = total

colors = ['#38bdf8', '#22c55e', '#f97316',
          '#a855f7', '#facc15', '#ef4444']
if sums:
    axes[0, 1].pie(
        sums.values(),
        labels=sums.keys(),
        autopct='%1.1f%%',
        colors=colors[:len(sums)],
        startangle=90
    )
axes[0, 1].set_title('Vehicle Type Distribution')

# ── Plot 3 — Green signal frequency per lane ─────────
if 'green_lane' in df.columns:
    lane_counts = df['green_lane'].value_counts()
    bar_colors  = ['#22c55e', '#38bdf8', '#f97316', '#a855f7']
    bars = axes[1, 0].bar(
        lane_counts.index,
        lane_counts.values,
        color=bar_colors[:len(lane_counts)]
    )
    axes[1, 0].set_title('Green Signal Frequency Per Lane')
    axes[1, 0].set_ylabel('Count')
    for bar, val in zip(bars, lane_counts.values):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val), ha='center', fontweight='bold'
        )

# ── Plot 4 — Inference latency ────────────────────────
if 'inference_latency_ms' in df.columns:
    axes[1, 1].plot(df['inference_latency_ms'].values,
                    color='#f97316', linewidth=1.5)
    avg = df['inference_latency_ms'].mean()
    axes[1, 1].axhline(
        y=avg, color='red', linestyle='--',
        label=f"Avg: {avg:.1f}ms"
    )
    axes[1, 1].set_title('Inference Latency (ms)')
    axes[1, 1].set_ylabel('Latency (ms)')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('logs', exist_ok=True)
plt.savefig('logs/analytics.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Analytics saved to logs/analytics.png")