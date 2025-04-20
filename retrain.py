import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from functions import extract_features, extract_ppg_signal

# ==== Feature Extraction (Expanded) ====
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

# ==== Synthetic Data with Save ====
def generate_better_mock_data_with_save(n=1000, save_path="synthetic_glucose_dataset.csv"):
    np.random.seed(42)
    n_diabetic = int(n * 0.5)
    n_normal = n - n_diabetic

    mean_normal = np.random.normal(loc=0.05, scale=0.1, size=n_normal)
    std_normal = np.random.normal(loc=1.0, scale=0.05, size=n_normal)
    hr_normal = np.random.normal(loc=70, scale=5, size=n_normal)
    entropy_normal = np.random.normal(loc=0.7, scale=0.1, size=n_normal)
    auc_normal = np.random.normal(loc=1.5, scale=0.3, size=n_normal)
    glucose_normal = 85 + 0.45 * hr_normal + 0.5 * mean_normal + 0.2 * entropy_normal + 0.1 * auc_normal + np.random.normal(0, 3, size=n_normal)

    mean_diabetic = np.random.normal(loc=0.15, scale=0.1, size=n_diabetic)
    std_diabetic = np.random.normal(loc=1.1, scale=0.08, size=n_diabetic)
    hr_diabetic = np.random.normal(loc=85, scale=7, size=n_diabetic)
    entropy_diabetic = np.random.normal(loc=0.85, scale=0.1, size=n_diabetic)
    auc_diabetic = np.random.normal(loc=2.5, scale=0.4, size=n_diabetic)
    glucose_diabetic = 150 + 0.5 * hr_diabetic + 0.6 * mean_diabetic + 0.25 * entropy_diabetic + 0.15 * auc_diabetic + np.random.normal(0, 4, size=n_diabetic)

    X = pd.DataFrame({
        "mean": np.concatenate([mean_normal, mean_diabetic]),
        "std": np.concatenate([std_normal, std_diabetic]),
        "heart_rate": np.concatenate([hr_normal, hr_diabetic]),
        "entropy": np.concatenate([entropy_normal, entropy_diabetic]),
        "auc": np.concatenate([auc_normal, auc_diabetic])
    })
    y = np.concatenate([glucose_normal, glucose_diabetic])

    X["glucose"] = y
    X.to_csv(save_path, index=False)

    return X.drop(columns="glucose"), y

# ==== Train Model with Real Sample Injection ====
def train_model_with_better_data(real_video_path="samples/reference.mp4", true_glucose_value=176.0):
    X, y = generate_better_mock_data_with_save()

    real_signal = extract_ppg_signal(real_video_path)
    real_features = extract_features(real_signal)
    if not real_features.empty:
        X = pd.concat([X, real_features], ignore_index=True)
        y = np.append(y, true_glucose_value)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model trained with enhanced synthetic and real data. Test MSE:", mean_squared_error(y_test, model.predict(X_test)))
    return model


# ==== Main ====
if __name__ == "__main__":
    print("\n=== Simplified Non-Invasive Glucose Estimator ===")
    model = train_model_with_better_data()

    with open('model.pkl','wb') as f:
        pickle.dump(model,f)
    
    print("Model successfully trained and saved as 'model.pkl'")
