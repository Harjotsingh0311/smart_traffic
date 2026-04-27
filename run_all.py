# run_all.py
import subprocess
import sys
import time
import os

print("🚦 Starting Smart Traffic Management System...")
print("=" * 50)

# Start main detection system
print("\n[1/2] Starting Detection + Dashboard...")
detection = subprocess.Popen(
    [sys.executable, 'main.py'],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)
print("✅ Detection system started")

time.sleep(3)

# Start pygame simulation
print("\n[2/2] Starting Traffic Simulation...")
simulation = subprocess.Popen(
    [sys.executable, 'simulation.py'],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)
print("✅ Simulation started")

print("\n" + "=" * 50)
print("✅ ALL SYSTEMS RUNNING!")
print("=" * 50)
print("\n📺 Detection window — shows YOLO bounding boxes")
print("🎮 Simulation window — shows intersection")
print("\nPress Ctrl+C to stop all systems")

try:
    detection.wait()
except KeyboardInterrupt:
    print("\n🛑 Stopping all systems...")
    detection.terminate()
    simulation.terminate()
    print("✅ All stopped")