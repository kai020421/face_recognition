
#  Student Face Recognition System

This is a Python-based **Student Face Recognition System** using **OpenCV**, **HOG**, **LBP**, and **Logistic Regression / Random Forest**. The goal is to detect faces in real-time or static images and accurately recognize the student using machine learning models trained on extracted features.

---

##  Features

-  Upload image or use webcam for face recognition
-  Comprehensive EDA including:
  - Image size, aspect ratio, brightness, blur score
  - Visualizations per student ID
-  Trained on facial features using:
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Patterns)
-  Visual explanation of model learning (PCA, confidence scores)
-  Model choices:
  - Decision Tree (initial)
  - Logistic Regression / Random Forest (final)
-  GUI built with Tkinter

---

##  Project Structure

```
face-recognition/
│
├── data/                       # Student face images
├── face_model.pkl              # Trained model
├── label_encoder.pkl           # Saved label encoder
├── gui.py                      # GUI file for recognition
├── face_recognition.ipynb      # Notebook for EDA & visualizations
├── report.pdf                  # IEEE format Project Report
└── README.md                   # Project documentation (this file)
```

---

##  Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/duaannaz/face-recognition.git
cd face-recognition
```

2. **Create a virtual environment**

```bash
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install manually:
```bash
pip install opencv-python scikit-learn numpy matplotlib joblib scikit-image pillow
```

---

## Training the Model

To retrain your model or test different classifiers, run:

```python
# In model_trainin.ipynb
# You can experiment with DecisionTreeClassifier or RandomForestClassifier
```

Your model will be saved as `face_model.pkl` and label encoder as `label_encoder.pkl`.

---

## Running the GUI

Make sure your model is trained and saved. Then run:

```bash
python gui.py
```

From the GUI:
- Click **"Upload Image"** to recognize a face from an image
- Click **"Use Webcam"** to perform real-time recognition
- If the face is unknown or confidence is low (less than 0.5), system will display **"Unknown"**

---


## Project Contributors

**Duaa Naz** | **Hadia Sajid** | **Huzaifa Awais** | **Kashaf Ansari**

---

## Note

- This project is built for educational purposes and to demonstrates the understanding of key CT (Computational Thinking) pillars — Decomposition, Pattern Recognition, Abstraction, and Algorithmic Thinking — across every stage, from image preprocessing and feature extraction to model training and GUI integration. .
- It uses basic classifiers and feature extraction — no deep learning models involved.

---

## If you found this useful

Give a ⭐️ on [GitHub](https://github.com/duaannaz/face-recognition)!
