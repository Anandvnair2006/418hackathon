from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
import pandas as pd
import numpy as np
import cv2
import os


# ==== Enhanced Signal Extraction with Better Noise Reduction ====
def butter_bandpass_filter(data, lowcut=0.7, highcut=3.5, fs=30.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_ppg_signal(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Video file '{video_path}' not found.")
        return np.array([])

    cap = cv2.VideoCapture(video_path)
    ppg_signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        if h < 300 or w < 300:
            continue

        roi = frame[h//3:2*h//3, w//3:2*w//3]
        red = roi[:, :, 2].astype(np.float32)
        green = roi[:, :, 1].astype(np.float32)
        blue = roi[:, :, 0].astype(np.float32)

        composite = 0.2989 * red + 0.5870 * green + 0.1140 * blue
        brightness = np.mean(composite)
        ppg_signal.append(brightness)

    cap.release()
    ppg_signal = np.array(ppg_signal)

    if len(ppg_signal) < 30:
        return np.array([])

    ppg_signal -= np.mean(ppg_signal)
    ppg_signal /= np.std(ppg_signal)
    filtered_signal = butter_bandpass_filter(ppg_signal)
    smoothed_signal = savgol_filter(filtered_signal, 31, 3)

    return smoothed_signal


def extract_features(signal):
    if len(signal) == 0:
        return pd.DataFrame()

    peaks, _ = find_peaks(signal, distance=20, prominence=0.5)
    if len(peaks) < 2:
        return pd.DataFrame()

    heart_rate = len(peaks) / (len(signal) / 30.0) * 60
    entropy = -np.sum((p := np.histogram(signal, bins=30, density=True)[0]) * np.log(p + 1e-10))
    auc = np.trapz(np.abs(signal))

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "heart_rate": heart_rate,
        "entropy": entropy,
        "auc": auc
    }
    return pd.DataFrame([features])