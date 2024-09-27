from flask import Flask, render_template, request,send_from_directory
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('brain_disease_classification_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image

# Function to classify the image
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_label = np.argmax(predictions)
    if class_label == 0:
        return "Mild Demented"
    elif class_label == 1:
        return "Non Demented"
    elif class_label == 2:
        return "Very Mild Demented"
    else:
        return "Unknown"


import numpy as np
import cv2

def is_mri_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection to detect features
    edges = cv2.Canny(gray_image, 100, 200)
    
    # Calculate the percentage of edge pixels in the image
    total_pixels = image.shape[0] * image.shape[1]
    edge_pixels = np.count_nonzero(edges)
    edge_pixel_percentage = (edge_pixels / total_pixels) * 100
    
    # MRI images typically have low edge content compared to non-MRI images
    # Set a threshold to classify the image as MRI or non-MRI
    if edge_pixel_percentage < 1.5:  # Adjust threshold as needed
        return False
    else:
        return True


import base64
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route('/upload_image/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('index.html', result="No image file provided.")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('upload.html', result="No image selected.")
    
    # Read and decode the uploaded image
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Check if the uploaded file is an MRI image
    if image is not None and is_mri_image(image):
        _, img_encoded = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(img_encoded).decode('utf-8')
        classification_result = classify_image(image)
    else:
        image_base64= None
        classification_result="Error! Please upload an MRI image."
    
    return render_template('template.html', result=classification_result,image=image_base64)
    
if __name__ == '__main__':
    app.run(debug=True)
