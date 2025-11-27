import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from deepface import DeepFace

# Set the paths for the dataset
dataset_path = "/path/to/dataset"

# Initialize DeepFace model (Facenet512)
model = DeepFace.build_model("Facenet")

# Collect image paths and labels
image_paths = []
labels = []
class_folders = [subfolder for subfolder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subfolder))]
for class_folder in class_folders:
    class_path = os.path.join(dataset_path, class_folder)
    image_files = [file for file in os.listdir(class_path) if file.endswith(".jpg") or file.endswith(".png")]
    image_paths.extend([os.path.join(class_path, img) for img in image_files])
    labels.extend([class_folder] * len(image_files))

# Encode the labels as integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Compute similarity scores
similarities = []
for i in range(len(X_test)):
    img1 = cv2.imread(X_test[i])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    for j in range(i+1, len(X_test)):
        img2 = cv2.imread(X_test[j])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Calculate similarity score using cosine distance
        score = DeepFace.distance([img1], [img2], model_name=model, distance_metric="cosine")
        similarities.append(score[0][0])

# Compute ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, similarities)
auc_score = roc_auc_score(y_test, similarities)

# Plot ROC curve
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
