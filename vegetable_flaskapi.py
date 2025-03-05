from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import io
import joblib
from PIL import Image

# Load the trained model correctly
model = joblib.load("veggies_predictor.h5")

# Define class labels (Update based on your model)
class_map = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli',
             5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber',
             10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}

# Initialize Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])  #, methods=['POST']
def predict():
    print("Predict called")
    
    # Check if image file is sent
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Read image from request
    file = request.files['file']

    contents = file.read()
    # Convert image to numpy array
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Preprocess image
    image = cv2.resize(image, (224, 224))
    image = image.reshape(1, 224, 224, 3)  # Normalize the image

    # Make prediction
    prediction = np.argmax(model.predict(image))
    predicted_class = class_map[prediction]
    print("Predicted class:", predicted_class)

    return jsonify({"prediction": predicted_class})

# Run the Flask app
if __name__ == "__main__":
    app.run(port=8000, debug=True)


# import h5py

# try:
#     with h5py.File("e:/api/vegetables_predictor.pkl", "r") as f:
#         print("File is a valid .h5 file")
# except Exception as e:
#     print("Error:", e)