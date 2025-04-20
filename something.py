from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import cv2
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
import pickle

app = Flask(__name__)
CORS(app)

# ===== Load model =====
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ===== Signal extraction =====
def butter_bandpass_filter(data, lowcut=0.7, highcut=3.5, fs=30.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_ppg_signal(video_path):
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
    return savgol_filter(filtered_signal, 31, 3)

# ===== Feature extraction =====
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

# ===== API Route =====
@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    video_path = "uploaded_video.mp4"
    video.save(video_path)

    signal = extract_ppg_signal(video_path)
    features = extract_features(signal)

    os.remove(video_path)

    if features.empty:
        return jsonify({"error": "Could not extract features. Try a better video."}), 400

    glucose = model.predict(features)[0]

    if glucose < 70:
        message = "🔻 Hypoglycemia alert! Grab a candy!"
    elif glucose > 140:
        message = "🔺 High sugar alert! Maybe skip that donut?"
    else:
        message = "✅ Sweet spot! You're in the normal range."

    return jsonify({"glucose": float(glucose), "message": message})

# ===== Frontend Route =====
@app.route("/", methods=["GET"])
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title> Gluco Vision</title>
  <style>
    body {
      background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
      font-family: 'Comic Sans MS', cursive, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }
    .sugar-box {
      background: #fff5f8;
      padding: 2rem;
      border-radius: 20px;
      box-shadow: 0 12px 25px rgba(255, 105, 180, 0.3);
      text-align: center;
      max-width: 400px;
      width: 90%;
      border: 4px dashed #ff69b4;
    }
    h1 {
      color: #ff4081;
      font-size: 2em;
      margin-bottom: 0.5rem;
    }
    input[type="file"] {
      margin: 1rem 0;
    }
    button {
      background-color: #ff69b4;
      color: white;
      border: none;
      padding: 0.7rem 1.5rem;
      font-size: 1rem;
      border-radius: 12px;
      cursor: pointer;
      transition: background-color 0.3s;
    }
    button:hover {
      background-color: #ff1493;
    }
    #result {
      margin-top: 1.5rem;
      font-size: 1.1rem;
      color: #d63384;
    }
  </style>
</head>
<body>
  <div class="sugar-box">
    <h1>🍬 Gluco Vision</h1>
    <p>Drop your fingertip video and let’s find out how sweet you are! 🍭</p>
    <input type="file" id="videoInput" accept=".mp4"/>
    <button onclick="predictGlucose()">Check My Sugar! 🍓</button>
    <div id="result"></div>
  </div>
  <script>
    function predictGlucose() {
      const videoFile = document.getElementById("videoInput").files[0];
      const resultDiv = document.getElementById("result");

      if (!videoFile) {
        resultDiv.innerHTML = "❌ Oops! Please upload your sugar video 🍩";
        return;
      }

      const formData = new FormData();
      formData.append("video", videoFile);

      resultDiv.innerHTML = "⏳ Crunching your sugar vibes... 🍬";

      fetch("/predict", {
        method: "POST",
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          if (data.error) {
            resultDiv.innerHTML = "❌ " + data.error;
          } else {
            resultDiv.innerHTML = `
              🍭 Your Glucose Level: <strong>${data.glucose.toFixed(1)} mg/dL</strong><br/>
              ${data.message}
            `;
          }
        })
        .catch(error => {
          console.error("Error:", error);
          resultDiv.innerHTML = "🍫 Something went wrong. Sugar crash!";
        });
    }
  </script>
</body>
</html>
    """)

# ===== Run the app =====
if __name__ == "__main__":
    app.run(debug=True)
