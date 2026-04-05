"""
ECG Heartbeat Classifier - FINAL (Normalized Excel Fixed)
Author: Hashmi
"""

import serial
import numpy as np
import joblib
import pandas as pd
import time
import sys
import os
import threading
from datetime import datetime

# ==================== CONFIGURATION ====================
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
SAMPLE_SIZE = 187

MODEL_PATH = r"C:\Users\htnha\Desktop\Ai project\newmodel.pkl"
SCALER_PATH = r"C:\Users\htnha\Desktop\Ai project\scaler.pkl"

EXCEL_FOLDER = r"C:\Users\htnha\Desktop\Ai project\ECG_Logs"

CLASS_LABELS = {
    0: 'Normal (N)',
    1: 'Supraventricular (S)',
    2: 'Ventricular (V)',
    3: 'Fusion (F)',
    4: 'Unknown (Q)'
}

stop_flag = False

# ==================== EXCEL SAVER ====================
class ExcelSaver:
    def __init__(self):
        self.rows = []

        if not os.path.exists(EXCEL_FOLDER):
            os.makedirs(EXCEL_FOLDER)

    def add(self, packet_no, normalized_values):
        self.rows.append([packet_no] + normalized_values.tolist())

    def save(self):
        if not self.rows:
            print("❌ No data to save")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(EXCEL_FOLDER, f"Normalized_ECG_{ts}.xlsx")

        columns = ["Packet"] + [f"S{i+1}" for i in range(SAMPLE_SIZE)]
        df = pd.DataFrame(self.rows, columns=columns)
        df.to_excel(path, index=False)

        print("\n✅ Excel Saved Successfully")
        print(f"📁 File: {path}")
        print(f"📊 Rows: {len(df)} | Columns: {len(df.columns)}")

# ==================== ECG PROCESSOR ====================
class ECGProcessor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    def preprocess(self, raw):
        x = np.array(raw, dtype=np.float32)

        # 1️⃣ DC removal
        x = x - np.mean(x)

        # 2️⃣ MIT-BIH normalization (-1 to +1)
        max_val = np.max(np.abs(x))
        if max_val > 0:
            x = x / max_val

        normalized_only = x.copy()   # ✅ THIS is saved to Excel

        x = x.reshape(1, -1)

        # 3️⃣ Scaler (model input only)
        if self.scaler is not None:
            x = self.scaler.transform(x)

        return x, normalized_only

    def classify(self, raw):
        x_model, x_norm = self.preprocess(raw)
        probs = self.model.predict_proba(x_model)[0]
        pred = np.argmax(probs)
        return pred, probs[pred], probs, x_norm

# ==================== SERIAL ====================
def connect():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print(f"✓ Connected to {SERIAL_PORT}")
    return ser

def read_line(ser):
    try:
        return ser.readline().decode().strip()
    except:
        return ""

# ==================== USER STOP THREAD ====================
def stop_listener():
    global stop_flag
    while True:
        if input().lower() in ['stop', 'exit', 'quit']:
            stop_flag = True
            break

# ==================== MAIN ====================
def main():
    processor = ECGProcessor()
    excel = ExcelSaver()
    ser = connect()

    threading.Thread(target=stop_listener, daemon=True).start()

    packet = 0

    try:
        while not stop_flag:
            line = read_line(ser)
            if not line.startswith("ECG:"):
                continue

            samples = [int(v) for v in line[4:].split(",") if v.strip().isdigit()]
            if len(samples) != SAMPLE_SIZE:
                continue

            packet += 1

            pred, conf, probs, norm = processor.classify(samples)

            print(f"\n[{packet}] ECG Packet Received | Samples: {len(samples)}")
            print(f"🔍 RESULT: {CLASS_LABELS[pred]}")
            print(f"📊 CONFIDENCE: {conf*100:.1f}%")

            print("Probabilities:")
            for i, p in enumerate(probs):
                print(f"  {CLASS_LABELS[i]}: {p*100:5.1f}%")

            print(f"📐 Normalized Min={norm.min():.6f}, Max={norm.max():.6f}")

            excel.add(packet, norm)
            print("-" * 55)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        excel.save()
        print("\n🛑 Session Ended")

# ==================== ENTRY ====================
if __name__ == "__main__":
    print("="*60)
    print("ECG HEARTBEAT CLASSIFIER (FINAL)")
    print("Type 'stop' to save normalized ECG to Excel")
    print("="*60)
    main()
