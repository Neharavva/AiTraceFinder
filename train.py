import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_PATH = "dataset"
IMAGE_SIZE = 128
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

X = []
y = []

labels = os.listdir(DATASET_PATH)

for label in labels:
    label_path = os.path.join(DATASET_PATH, label)
    
    # Recursively walk through subdirectories
    for root, dirs, files in os.walk(label_path):
        for file in files:
            if not file.lower().endswith(ALLOWED_EXTENSIONS):
                continue

            img_path = os.path.join(root, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            X.append(img.flatten())
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total images loaded:", len(X))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("✅ Training completed")
print("✅ Model saved as model.pkl")
