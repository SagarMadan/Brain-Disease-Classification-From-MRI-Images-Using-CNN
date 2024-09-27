import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset and preprocess images
def load_dataset(data_dir):
    X = []
    y = []

    # Define the categories
    categories = ['MildDemented', 'NonDemented', 'VeryMildDemented']
    
    # Loop through the categories
    for i, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for file in os.listdir(category_path):
            image_path = os.path.join(category_path, file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = cv2.resize(image, (224, 224))  # Resize
            image = image / 255.0  # Normalize pixel values
            X.append(image)
            y.append(i)  # Append the category index as the label

    return np.array(X), np.array(y)

# Define CNN model
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 output classes: Mild, Non, Very Mild
    ])
    return model

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile the model
    model = create_model(input_shape=X_train[0].shape)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history=model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)
    
    # Save the trained model
    model.save('brain_disease_classification_model.h5')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    # Define the path to the dataset directory
    dataset_train_dir = os.path.join('dataset', 'train')

    # Load dataset and preprocess images
    X, y = load_dataset(dataset_train_dir)
    
    # Train the model
    train_model(X, y)
