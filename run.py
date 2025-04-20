from functions import extract_features, extract_ppg_signal
import pickle


def predict_glucose(video_path, model):
    signal = extract_ppg_signal(video_path)
    features = extract_features(signal)

    if features.empty:
        print("❌ Feature extraction failed. Cannot predict.")
        return

    prediction = model.predict(features)[0]
    print(f"🩸 Estimated Glucose: {prediction:.1f} mg/dL")
    if prediction < 70:
        print("🔻 Hypoglycemia risk - Consider consuming sugar")
    elif prediction > 140:
        print("🔺 Elevated glucose - Monitor or consult physician")
    else:
        print("✅ Normal glucose range")
    return prediction


if __name__=="__main__":
    # load
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    video_file = input("Enter path to fingertip video: ").strip('"')
    if video_file == "":
        print("Running Demo....")
        video_file = "samples/demo.mp4"
    
    predict_glucose(video_file, model)