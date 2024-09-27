import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# Function to extract features from MRI images
def extract_features_from_images(image_folder):
    features = []
    labels = []
    for folder in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder)
        if os.path.isdir(folder_path):
            label = folder
            for file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file)
                if image_path.endswith('.jpg') or image_path.endswith('.png'):
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    # Perform feature extraction from the image (e.g., using intensity histogram, texture features, etc.)
                    # Here, you can use various techniques to extract features from the image
                    # For simplicity, let's just use the mean intensity value as a feature
                    mean_intensity = np.mean(image)
                    features.append(mean_intensity)
                    labels.append(label)
    return features, labels

# Path to the dataset folder
dataset_folder = 'dataset'

# Extract features and labels from the training dataset
train_folder = os.path.join(dataset_folder, 'train')
train_features, train_labels = extract_features_from_images(train_folder)

# Extract features and labels from the test dataset
test_folder = os.path.join(dataset_folder, 'test')
test_features, test_labels = extract_features_from_images(test_folder)

# Convert labels to numerical values (e.g., age)
label_mapping = {'MildDemented': 1, 'NonDemented': 0, 'VeryMildDemented': 2}
train_labels = [label_mapping[label] for label in train_labels]
test_labels = [label_mapping[label] for label in test_labels]

# Convert lists to NumPy arrays
train_features = np.array(train_features).reshape(-1, 1)
test_features = np.array(test_features).reshape(-1, 1)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Train the Support Vector Regression (SVR) model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on the validation set
val_predictions = svm_model.predict(X_val)

# Evaluate the model
val_mae = mean_absolute_error(y_val, val_predictions)
print("Validation MAE:", val_mae)

# Predict on the test set
test_predictions = svm_model.predict(test_features)

# Evaluate the model on the test set
test_mae = mean_absolute_error(test_labels, test_predictions)
print("Test MAE:", test_mae)
