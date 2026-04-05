"""
ECG Heartbeat Classifier - FINAL MIT-BIH Compatible Version
Author: Hashmi
"""

import serial
import numpy as np
import joblib
import time
import sys
import os

# ==================== CONFIGURATION ====================
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
SAMPLE_SIZE = 187

MODEL_PATH = r"C:\Users\htnha\Desktop\Ai project\newmodel.pkl"
SCALER_PATH = r"C:\Users\htnha\Desktop\Ai project\scaler.pkl"

CLASS_LABELS = {
    0: 'Normal (N)',
    1: 'Supraventricular (S)',
    2: 'Ventricular (V)',
    3: 'Fusion (F)',
    4: 'Unknown (Q)'
}

# ==================== ECG PROCESSOR ====================
class ECGProcessor:
    def __init__(self):
        print("Loading model and scaler...")

        if not os.path.exists(MODEL_PATH):
            sys.exit("❌ Model file not found")

        self.model = joblib.load(MODEL_PATH)
        print("✓ Model loaded")

        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
            print("✓ Scaler loaded")
        else:
            self.scaler = None
            print("⚠ No scaler found (using manual normalization)")

    # ---------- MIT-BIH STYLE PREPROCESSING ----------
    def preprocess(self, raw_signal):
        if len(raw_signal) != SAMPLE_SIZE:
            return None

        x = np.array(raw_signal, dtype=np.float32)

        # 1️⃣ Remove DC offset
        x = x - np.mean(x)

        # 2️⃣ Amplitude normalization (MIT-BIH style)
        max_val = np.max(np.abs(x))
        if max_val > 0:
            x = x / max_val

        # 3️⃣ Reshape
        x = x.reshape(1, -1)

        # 4️⃣ Apply scaler if used during training
        if self.scaler is not None:
            x = self.scaler.transform(x)

        return x

    def classify(self, raw_signal):
        x = self.preprocess(raw_signal)
        if x is None:
            return None, None, None

        probs = self.model.predict_proba(x)[0]
        pred = np.argmax(probs)
        confidence = probs[pred]

        return pred, confidence, probs

# ==================== SERIAL FUNCTIONS ====================
def connect_to_esp32():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        print(f"✓ Connected to {SERIAL_PORT}")
        return ser
    except Exception as e:
        sys.exit(f"❌ Serial error: {e}")

def read_line(ser):
    try:
        return ser.readline().decode('utf-8', errors='ignore').strip()
    except:
        return ""

# ==================== MAIN LOOP ====================
def main():
    print("\nECG HEARTBEAT CLASSIFICATION SYSTEM\n")

    processor = ECGProcessor()
    ser = connect_to_esp32()

    packet_count = 0

    try:
        while True:
            line = read_line(ser)
            if not line:
                continue

            if line.startswith("ECG:"):
                packet_count += 1
                data = line.split("ECG:", 1)[1]

                try:
                    samples = [int(v) for v in data.split(",") if v.strip().isdigit()]
                except:
                    continue

                if len(samples) != SAMPLE_SIZE:
                    continue

                print(f"\n[{packet_count}] ECG Packet")
                print(f"Samples: {len(samples)}, Min: {min(samples)}, Max: {max(samples)}")

                pred, conf, probs = processor.classify(samples)

                if pred is not None:
                    print(f"\n🔍 RESULT: {CLASS_LABELS[pred]}")
                    print(f"📊 CONFIDENCE: {conf*100:.1f}%\n")

                    print("Probabilities:")
                    for i, p in enumerate(probs):
                        print(f"  {CLASS_LABELS[i]}: {p*100:5.1f}%")

                print("-" * 55)

    except KeyboardInterrupt:
        print("\nSession ended.")
    finally:
        ser.close()

# ==================== TEST MODE ====================
def test_with_synthetic_data():
    print("\nRunning synthetic ECG test...\n")
    processor = ECGProcessor()

    baseline = 1800
    ecg = []

    for i in range(SAMPLE_SIZE):
        if 90 <= i <= 110:
            ecg.append(baseline + np.random.randint(600, 900))
        elif 60 <= i <= 80:
            ecg.append(baseline + np.random.randint(150, 300))
        elif 120 <= i <= 140:
            ecg.append(baseline + np.random.randint(200, 350))
        else:
            ecg.append(baseline + np.random.randint(-50, 50))

    pred, conf, probs = processor.classify(ecg)

    print("Synthetic ECG Result:")
    print(f"Prediction: {CLASS_LABELS[pred]}")
    print(f"Confidence: {conf*100:.1f}%")

# ==================== ENTRY ====================
if __name__ == "__main__":
    test_with_synthetic_data()
    if input("\nStart real-time monitoring? (y/n): ").lower() == 'y':
        main()
