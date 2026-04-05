"""
ECG Real-Time Visualization + Normalization (0-1)
Author: Hashmi
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
import time

# ================= CONFIG =================
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
SAMPLE_SIZE = 187

# ================= NORMALIZATION 0-1 =================
def normalize_ecg(raw):
    x = np.array(raw, dtype=np.float32)
    x = x - np.min(x)          # Shift minimum to 0
    max_val = np.max(x)
    if max_val > 0:
        x = x / max_val        # Scale max to 1
    return x

# ================= MAIN =================
def main():
    print("\nConnecting to ESP32...\n")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)

    all_beats = []
    packet_count = 0

    plt.ion()  # Turn on interactive plotting
    fig, ax = plt.subplots(figsize=(12, 4))

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line.startswith("ECG:"):
                continue

            data = line.split("ECG:", 1)[1]

            try:
                samples = [int(v) for v in data.split(",") if v.strip().isdigit()]
            except:
                continue

            if len(samples) != SAMPLE_SIZE:
                continue

            packet_count += 1
            normalized = normalize_ecg(samples)
            all_beats.append(normalized)

            # Real-time plotting of current beat
            ax.clear()
            ax.plot(range(SAMPLE_SIZE), normalized, color='blue')
            ax.set_title(f"Real-Time ECG Beat #{packet_count}")
            ax.set_xlabel("Sample Number")
            ax.set_ylabel("Normalized Voltage (0-1)")
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            plt.pause(0.05)  # Small pause to update plot

    except KeyboardInterrupt:
        print("\n\nStopping real-time monitoring...")
    finally:
        ser.close()
        plt.ioff()  # Turn off interactive mode
        print(f"Total beats captured: {len(all_beats)}")

        if all_beats:
            # Plot all beats sequentially in one figure (stacked 0-1)
            plt.figure(figsize=(12, len(all_beats)*0.5 + 3))
            for idx, beat in enumerate(all_beats):
                plt.plot(range(SAMPLE_SIZE), beat + idx*1.2, color='blue')  # offset for stacking
            plt.title("Captured ECG Beats (Stacked View, 0-1)")
            plt.xlabel("Sample Number")
            plt.ylabel("Normalized Voltage + Offset")
            plt.grid(True)
            plt.show()
        else:
            print("No ECG data captured.")

# ================= RUN =================
if __name__ == "__main__":
    main()
