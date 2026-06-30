# 📸 Screen Photo Detector

An AI-powered computer vision system that detects whether an image is a **real-world photograph** or a **photograph of a digital screen (recapture attack)** using **MobileNetV3-Large** and **PyTorch**.

The project includes a complete end-to-end pipeline, from dataset collection and model training to deployment as a Flask web application with live webcam support.

---
# Demo

<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/a173a61d-2184-4786-8567-1cdb8a884c8e" />

<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/39108d93-db7f-4c78-8417-6fa8c02bf79a" />

---

# 🚀 Features

- 🤖 Binary Image Classification (Real vs Screen)
- 📷 Live Webcam Detection
- 📂 Image Upload
- 📸 Camera Capture
- 📊 Confidence Score
- ⚠️ Fraud Score
- 🌐 Flask Web Application
- 📱 Mobile-friendly MobileNetV3 Architecture
- ⚡ Real-time Predictions

---

# 🏗️ Project Flow

```text
                Dataset Collection
                       │
                       ▼
            Data Preprocessing & Augmentation
                       │
                       ▼
       Transfer Learning (MobileNetV3-Large)
                       │
                       ▼
                Model Fine-Tuning
                       │
                       ▼
             Save Trained Weights (.pth)
                       │
                       ▼
            Flask Backend (Inference API)
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
  Upload Image                Live Camera Capture
         │                           │
         └─────────────┬─────────────┘
                       ▼
             Image Preprocessing
                       ▼
            MobileNetV3 Prediction
                       ▼
       ┌────────────────────────────────┐
       │                                │
       │   Prediction (REAL / SCREEN)   │
       │   Confidence Score             │
       │   Fraud Score                  │
       └────────────────────────────────┘
```

---

# 📂 Project Structure

```
.
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── benchmark.py
├── dataset.py
├── requirements.txt
│
├── models/
│   └── mobilenet.py
│
├── templates/
│   └── index.html
│
├── weights/
│   └── best_model.pth
│
├── train/
├── val/
└── test/
```

---

# 🧠 Model Architecture

- **Backbone:** MobileNetV3-Large
- **Framework:** PyTorch
- **Transfer Learning**
- Binary Classification Head
- Softmax Output Layer

Classes:

- 🟢 REAL
- 🔴 SCREEN

---

# 📊 Results

| Metric | Value |
|---------|--------|
| Test Accuracy | **93.75%** |
| Latency | **~427 ms/image** |
| Device | Apple MacBook Air (M2) |
| Framework | PyTorch (MPS) |
| Cost per Image | **≈ $0 (On-device)** |

---

# 🌐 Web Application Workflow

1. User uploads an image or captures one using the webcam.
2. Flask receives the image.
3. The image is preprocessed (resize + normalization).
4. MobileNetV3-Large performs inference.
5. Softmax probabilities are computed.
6. The application displays:
   - Prediction
   - Confidence
   - Fraud Score
7. The analyzed image is shown alongside the prediction.

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Abhiman74/-screen-photo-detector.git
cd -screen-photo-detector
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Train the Model

```bash
python train.py
```

---

# 🔍 Predict an Image

```bash
python predict.py path/to/image.jpg
```

Example Output

```
Prediction : REAL
Confidence : 69.38%
Fraud Score : 0.3062
```

---

# 🌍 Run the Web App

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 📈 Future Improvements

- Increase dataset size
- More lighting conditions
- More screen types
- Quantized MobileNet for faster inference
- ONNX / TensorRT deployment
- Android/iOS integration

---

# 👨‍💻 Author

**Abhiman Singh**

B.Tech Computer Science Engineering  
Bennett University

GitHub: https://github.com/Abhiman74

---

# 📄 License

Developed for educational purposes as part of a Machine Learning assignment.
