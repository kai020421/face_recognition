import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern

# Load Model and Label Encoder
model = joblib.load('face_model.pkl')
le = joblib.load('label_encoder.pkl')

THRESHOLD = 0.5  # below this probability, we’ll label as Unknown

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# LBP Parameters
radius = 1
n_points = 8 * radius

# Feature Extractor (HOG + LBP)
def extract_features(gray_face):
    face = cv2.resize(gray_face, (100, 100)).astype('float32') / 255.0

    hog_feat = hog(face, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(face, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points+2))
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    return np.hstack([hog_feat, lbp_hist])

# Predict face in an image
def predict_image(img, draw_rectangle=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return img, "No face detected"

    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    face = gray[y:y+h, x:x+w]
    if np.var(face) < 100:
        return img, "Face too blurry"

    features = extract_features(face).reshape(1, -1)
    proba = model.predict_proba(features)[0]
    best_idx = np.argmax(proba)
    confidence = proba[best_idx]

    if confidence < THRESHOLD:
        label = "Unknown"
    else:
        label = le.inverse_transform([best_idx])[0]
        label += f" ({confidence*100:.1f}%)"

    if draw_rectangle:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return img, label

# Upload Image
def upload_image():
    path = filedialog.askopenfilename()
    if not path:
        return
    img = cv2.imread(path)
    annotated_img, result = predict_image(img)
    display_image(annotated_img, result)

# Webcam Capture
def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, result = predict_image(frame)
        cv2.imshow("Live Recognition - Press Q to Quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Display Image + Prediction
def display_image(img, result):
    img = cv2.resize(img, (400, 400))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im)

    label_img.config(image=imgtk)
    label_img.image = imgtk
    label_result.config(text=f"Prediction: {result}")

# GUI Setup
root = tk.Tk()
root.title("Student Face Recognition")
root.geometry("500x550")

btn_upload = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 12))
btn_upload.pack(pady=10)

btn_webcam = tk.Button(root, text="Use Webcam", command=start_webcam, font=("Arial", 12))
btn_webcam.pack(pady=10)

label_img = tk.Label(root)
label_img.pack(pady=10)

label_result = tk.Label(root, text="", font=("Arial", 14))
label_result.pack(pady=10)

root.mainloop()