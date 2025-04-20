# Gluco Vision

## 📦 Features

- Extracts PPG signals from video using color channels and region of interest.
- Applies **Butterworth bandpass** and **Savitzky-Golay smoothing** for noise reduction.
- Computes physiological features like **heart rate**, **entropy**, and **signal area**.
- Generates **synthetic glucose data** (normal + diabetic distributions).
- Combines real and synthetic data to train a **Gradient Boosting Regressor**.
- Outputs glucose prediction with simple risk level warnings.


##Install Dependencies 
```bash
pip install opencv-python numpy pandas scipy scikit-learn

##📈 Example Output
✅ Model trained with enhanced synthetic and real data. Test MSE: 92.7
🩸 Estimated Glucose: 142.3 mg/dL
🔺 Elevated glucose - Monitor or consult physician

##🧪 Future Improvements
Add face detection and tracking

Support multiple video formats

Integrate with real-world datasets

Improve model performance with neural networks

🙏 Acknowledgements
This project was developed with the support of the following tools and research contributions:

OpenAI's ChatGPT – Assisted in brainstorming, refining code structure, signal processing techniques, and improving clarity and documentation.

Marjan Jusev et al. – The research work titled "Non-invasive Glucose Estimation via Video-based Photoplethysmography" provided valuable insights into the correlation between PPG signal dynamics and glucose levels. Their findings significantly influenced the signal feature extraction and modeling strategies used in this project.

We gratefully acknowledge their contributions to the advancement of non-invasive health monitoring technologies.
# 418hackathon
creation of a website that uses video of your fingertips to find your glucose level.
